import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def prototype_contrast_transform(
    x_all,
    x_train,
    y_train,
    gamma=0.15,
    use_adaptive_weights=True,
    distance_metric='euclidean',
):
    """Apply prototype-guided attraction and repulsion to all features."""
    # Build one prototype per support class.
    class_prototypes = {}
    for class_id in np.unique(y_train):
        class_mask = y_train == class_id
        class_samples = x_train[class_mask]
        class_prototypes[class_id] = np.mean(class_samples, axis=0)
    x_transformed = x_all.copy()
    n_train = len(x_train)
    for i, x_i in enumerate(x_all):
        distances_to_prototypes = {}
        for class_id, prototype in class_prototypes.items():
            if distance_metric == 'cosine':
                cos_sim = np.dot(x_i, prototype) / (np.linalg.norm(x_i) * np.linalg.norm(prototype) + 1e-12)
                distances_to_prototypes[class_id] = 1.0 - cos_sim
            else:
                distances_to_prototypes[class_id] = np.linalg.norm(x_i - prototype)
        nearest_class = min(distances_to_prototypes, key=distances_to_prototypes.get)
        if i < n_train:
            target_class = y_train[i]
        else:
            target_class = nearest_class
        attraction = class_prototypes[target_class] - x_i

        # Push the sample away from every non-target prototype.
        repulsion = np.zeros_like(x_i)
        other_distances = [dist for cls, dist in distances_to_prototypes.items() if cls != target_class]
        if len(other_distances) > 0:
            min_other_distance = min(other_distances)
            for class_id, prototype in class_prototypes.items():
                if class_id != target_class:
                    to_other = x_i - prototype
                    other_distance = distances_to_prototypes[class_id]
                    if use_adaptive_weights:
                        relative_distance = other_distance / (min_other_distance + 1e-06)
                        repel_weight = 1.0 / (relative_distance + 1e-06)
                    else:
                        repel_weight = 1.0 / (other_distance + 1e-06)
                    repulsion += repel_weight * to_other
        total_transform = attraction + repulsion
        transform_magnitude = np.linalg.norm(total_transform)
        original_magnitude = np.linalg.norm(x_i)
        if transform_magnitude > gamma * original_magnitude:
            total_transform = total_transform * (gamma * original_magnitude) / transform_magnitude
        x_transformed[i] = x_i + total_transform
    return x_transformed

class DPGS:
    """DPGS evaluator."""

    def __init__(self, model, device, log_file, args):
        self.device = device
        self.beta = args.beta
        self.balanced = args.balanced
        self.n_shot = args.shot
        self.k_min = args.k_min
        self.k_max = args.k_max

        self.gamma = args.gamma

        # Soft k-means controls.
        self.T_km = args.T_km
        self.nIter_km = args.nIter_km

        
        self.alpha_pslp = args.alpha_pslp
        self.beta_pslp = args.beta_pslp
        self.k_graph = args.k_graph
        self.ot_epsilon = args.ot_epsilon
        self.ot_max_iters = args.ot_max_iters
        self.use_pslp = self.balanced == 'dirichlet' and self.n_shot > 1


        # Task-specific FiLM adaptation.
        self.film_hidden_dim = args.film_hidden_dim
        self.film_lr = args.film_lr
        self.film_inner_steps = args.film_inner_steps
        self.lambda_ce = args.lambda_ce
        self.lambda_cluster = args.lambda_cluster
        self.lambda_inter_sep = args.lambda_inter_sep
        self.lambda_intra_compact = args.lambda_intra_compact
        self.cluster_prob_thresh = args.cluster_prob_thresh
        self.cluster_rho_thresh_percentile = args.cluster_rho_thresh_percentile

        self.cond_mlp = None
        self.transform_opt = None

    def compute_prototype_ce_loss(self, x_samples, y_samples, mus):
        """Compute prototype-based cross-entropy loss."""
        dist = torch.cdist(x_samples, mus, p=2)
        logits = -dist
        loss = F.cross_entropy(logits, y_samples)
        return loss

    def compute_cluster_structure_loss(
        self,
        x_query,
        y_pseudo,
        prob_max,
        rho_query,
        mus,
        prob_thresh=0.9,
        rho_thresh_percentile=50,
    ):
        """Compute the density-weighted cluster-structure loss."""
        N_query = x_query.shape[0]
        if N_query == 0:
            return (
                torch.tensor(0.0, device=x_query.device),
                torch.zeros(N_query, dtype=torch.bool, device=x_query.device),
            )
        if not rho_query.is_floating_point():
            rho_query = rho_query.float()
        rho_min = rho_query.min()
        rho_max = rho_query.max()
        if rho_max - rho_min > 1e-08:
            rho_norm_global = (rho_query - rho_min) / (rho_max - rho_min)
        else:
            rho_norm_global = torch.ones_like(rho_query)
        rho_threshold = torch.quantile(rho_query, rho_thresh_percentile / 100.0)
        rho_mask = rho_query >= rho_threshold
        prob_mask = prob_max >= prob_thresh
        high_confidence_mask = prob_mask & rho_mask
        if high_confidence_mask.sum() == 0:
            return (torch.tensor(0.0, device=x_query.device), high_confidence_mask)
        x_high = x_query[high_confidence_mask]
        y_high = y_pseudo[high_confidence_mask]
        prob_high = prob_max[high_confidence_mask]
        rho_high_norm = rho_norm_global[high_confidence_mask]
        w_high = rho_high_norm * prob_high
        w_high = torch.clamp(w_high, min=0.0)
        distances = torch.cdist(x_high, mus, p=2)
        logits = -distances
        log_probs = F.log_softmax(logits, dim=1)
        y_high_expanded = y_high.unsqueeze(1)
        log_p_target = torch.gather(log_probs, dim=1, index=y_high_expanded).squeeze(1)
        weighted_loss_sum = -(log_p_target * w_high).sum()
        W_total = w_high.sum()
        if W_total < 1e-06:
            return (torch.tensor(0.0, device=x_query.device), high_confidence_mask)
        loss = weighted_loss_sum / W_total
        return (loss, high_confidence_mask)

    def compute_inter_class_separation_loss(self, mus):
        """Encourage separation between class prototypes."""
        n_ways = mus.shape[0]
        dist_matrix = torch.cdist(mus, mus, p=2)
        mask = torch.triu(torch.ones(n_ways, n_ways, device=self.device), diagonal=1).bool()
        inter_distances = dist_matrix[mask]
        margin = 1
        loss = F.relu(margin - inter_distances).mean()
        return loss

    def compute_intra_class_compactness_loss(self, x_samples, y_samples, mus):
        """Encourage samples to stay close to their assigned prototypes."""
        assigned_mus = mus[y_samples]
        distances = torch.norm(x_samples - assigned_mus, p=2, dim=1)
        return distances.mean()

    def initialize_film_modules(self, feature_dim):
        """Initialize the task-specific FiLM transformation."""
        class FilmMLP(nn.Module):

            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
                with torch.no_grad():
                    self.net[-1].weight.mul_(0.01)
                    self.net[-1].bias.zero_()

            def forward(self, cond):
                out = self.net(cond)
                gamma_raw, beta_raw = out.chunk(2, dim=-1)
                gamma = torch.sigmoid(gamma_raw) * 1.0 + 0.5
                return (gamma, beta_raw)
        self.cond_mlp = FilmMLP(
            input_dim=2 * feature_dim,
            hidden_dim=self.film_hidden_dim,
            output_dim=2 * feature_dim,
        ).to(self.device)
        self.transform_opt = torch.optim.Adam(list(self.cond_mlp.parameters()), lr=self.film_lr)

    def calculate_density_and_distance(self, x_all):
        """Compute local density, pairwise distance, and density links."""
        n_total = len(x_all)
        distances = squareform(pdist(x_all, metric='euclidean'))
        global_k = min(15, n_total // 10)
        global_rho = np.zeros(n_total)
        for i in range(n_total):
            neighbors_distances = distances[i]
            sorted_dist = np.sort(neighbors_distances)[1:global_k + 1]
            global_rho[i] = global_k / (np.sum(sorted_dist) + 1e-06)
        rho_percentiles = np.percentile(global_rho, [25, 75])
        adaptive_k_values = np.zeros(n_total, dtype=int)
        for i in range(n_total):
            if global_rho[i] > rho_percentiles[1]:
                adaptive_k_values[i] = self.k_min
            elif global_rho[i] < rho_percentiles[0]:
                adaptive_k_values[i] = self.k_max
            else:
                ratio = (global_rho[i] - rho_percentiles[0]) / (rho_percentiles[1] - rho_percentiles[0])
                adaptive_k_values[i] = int(self.k_max - ratio * (self.k_max - self.k_min))
        self.knn_neighbors = {}
        for i in range(n_total):
            neighbors_distances = distances[i]
            sorted_indices = np.argsort(neighbors_distances)
            k_i = adaptive_k_values[i]
            self.knn_neighbors[i] = sorted_indices[1:k_i + 1]
        rho = np.zeros(n_total)
        for i in range(n_total):
            neighbors_distances = distances[i]
            k_i = adaptive_k_values[i]
            rho[i] = k_i / (np.sum(np.sort(neighbors_distances)[1:k_i + 1]) + 1e-06)
        nearest_higher_density = np.zeros(n_total, dtype=int) - 1
        sorted_indices = np.argsort(-rho)
        for idx, i in enumerate(sorted_indices):
            if idx == 0:
                nearest_higher_density[i] = -1
            else:
                higher_density_indices = sorted_indices[:idx]
                min_dist = float('inf')
                min_idx = -1
                for j in higher_density_indices:
                    if distances[i][j] < min_dist:
                        min_dist = distances[i][j]
                        min_idx = j
                nearest_higher_density[i] = min_idx
        return (rho, distances, nearest_higher_density)

    def find_candidate_centers(
        self,
        support_idx,
        nearest_higher_density,
        rho,
        y_train,
        target_class,
        n_train,
        n_total,
        x_train,
        x_test,
        class_centroids,
        ambig_window=0.03,
        use_cosine=False,
    ):
        """Extend a density chain from one support sample."""
        chain = [support_idx]
        current_idx = support_idx
        while True:
            next_idx = nearest_higher_density[current_idx]
            candidates = []
            if next_idx != -1 and next_idx < n_total and (next_idx in self.knn_neighbors[current_idx]):
                candidates.append(next_idx)
            for neighbor_idx in self.knn_neighbors[current_idx]:
                if neighbor_idx == next_idx:
                    continue
                if neighbor_idx >= n_total:
                    continue
                if rho[neighbor_idx] <= rho[current_idx]:
                    continue
                candidates.append(neighbor_idx)
            if not candidates:
                break
            extended = False
            for candidate_idx in candidates:
                if candidate_idx < n_train:
                    if y_train[candidate_idx] != target_class:
                        continue
                if class_centroids is not None:
                    feat = x_train[candidate_idx] if candidate_idx < n_train else x_test[candidate_idx - n_train]
                    if use_cosine:

                        def cos_sim(a, b):
                            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                        d_target = 1.0 - cos_sim(feat, class_centroids[target_class])
                        d_other_min = min(
                            1.0 - cos_sim(feat, cent)
                            for cls, cent in class_centroids.items()
                            if cls != target_class
                        )
                    else:
                        d_target = np.linalg.norm(feat - class_centroids[target_class])
                        d_other_min = min(
                            np.linalg.norm(feat - cent)
                            for cls, cent in class_centroids.items()
                            if cls != target_class
                        )
                    denom = d_target + d_other_min
                    ratio = d_target / denom if denom > 1e-12 else 1.0
                    lower, upper = (0.5 - ambig_window / 2, 0.5 + ambig_window / 2)
                    if lower <= ratio <= upper:
                        continue
                    if d_target > d_other_min:
                        continue
                chain.append(candidate_idx)
                current_idx = candidate_idx
                extended = True
                break
            if not extended:
                break
        return chain

    def assign_chain_labels_and_find_centers(
        self,
        x_train,
        y_train,
        x_test,
        rho,
        nearest_higher_density,
        distances,
        ambig_window=0.03,
        use_cosine=False,
    ):
        """Build class-wise density chains and collect their centers."""
        n_train, n_test = (len(x_train), len(x_test))
        n_total = n_train + n_test
        class_centroids = {cls: np.mean(x_train[y_train == cls], axis=0) for cls in np.unique(y_train)}
        all_labels = np.full(n_total, -1, dtype=int)
        all_labels[:n_train] = y_train
        all_centers, class_center_map = (set(), {})
        for class_label in np.unique(y_train):
            class_indices = np.where(y_train == class_label)[0]
            class_centers, = (set(),)
            for support_idx in class_indices:
                density_chain = self.find_candidate_centers(
                    support_idx,
                    nearest_higher_density,
                    rho,
                    y_train,
                    class_label,
                    n_train,
                    n_total,
                    x_train,
                    x_test,
                    class_centroids,
                    ambig_window=ambig_window,
                    use_cosine=use_cosine,
                )
                for idx in density_chain:
                    if all_labels[idx] == -1:
                        all_labels[idx] = class_label
                center_idx = density_chain[-1]
                if len(density_chain) > 1:
                    class_centers.add(center_idx)
                    all_centers.add(center_idx)
            class_center_map[class_label] = list(class_centers)
        return (all_centers, class_center_map, all_labels)

    def build_propagation_matrix(self, x_all, distances, k=30, alpha=0.5):
        """Build the graph-based label-propagation matrix."""
        n_total = len(x_all)
        W = np.zeros((n_total, n_total))
        for i in range(n_total):
            neighbors_idx = np.argsort(distances[i])[1:k + 1]
            sigma = np.mean(distances[i][neighbors_idx])
            for j in neighbors_idx:
                weight = np.exp(-distances[i][j] ** 2 / (2 * sigma ** 2))
                W[i, j] = weight
                W[j, i] = weight
        D = np.diag(W.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-08))
        L = np.eye(n_total) - D_inv_sqrt @ W @ D_inv_sqrt
        I = np.eye(n_total)
        try:
            P = np.linalg.inv(I - alpha * L)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(I - alpha * L)
        P_torch = torch.from_numpy(P).float()
        return P_torch

    def compute_optimal_transport(
        self,
        M,
        n_support,
        support_labels,
        labeled_indices=None,
        epsilon=0.001,
        max_iters=1000,
    ):
        """Balance soft assignments with Sinkhorn-Knopp iterations."""
        n_total, n_ways = M.shape
        device = M.device
        support_counts = torch.zeros(n_ways, device=device)
        for i in range(n_support):
            support_counts[support_labels[i]] += 1.0
        if labeled_indices is not None and len(labeled_indices) > n_support:
            center_indices = labeled_indices[n_support:]
            for idx in center_indices:
                if idx < n_total:
                    class_idx = M[idx].argmax()
                    support_counts[class_idx] += 1.0
        n_query = n_total - len(labeled_indices) if labeled_indices is not None else n_total - n_support
        c = support_counts + n_query / n_ways
        r = torch.ones(n_total, device=device)
        P = M.clone()
        P = P / (P.sum(dim=1, keepdim=True) + 1e-10)
        for _ in range(max_iters):
            P_old = P.clone()
            row_sum = P.sum(dim=1, keepdim=True)
            P = P * (r.unsqueeze(1) / (row_sum + 1e-10))
            col_sum = P.sum(dim=0, keepdim=True)
            P = P * (c.unsqueeze(0) / (col_sum + 1e-10))
            if labeled_indices is not None:
                for idx in labeled_indices:
                    if idx < n_total:
                        P[idx] = M[idx]
            else:
                P[:n_support] = 0.0
                P[:n_support].scatter_(1, support_labels.unsqueeze(1), 1.0)
            row_error = torch.max(torch.abs(P.sum(dim=1) - r))
            col_error = torch.max(torch.abs(P.sum(dim=0) - c))
            change = torch.max(torch.abs(P - P_old))
            if row_error < epsilon and col_error < epsilon and (change < epsilon):
                break
        return P

    def assign_unlabeled_points(
        self,
        x_train,
        y_train,
        x_test,
        all_labels,
        distances,
        n_train,
        all_centers,
        class_center_map,
    ):
        """Assign query labels with graph propagation or soft k-means."""
        labeled_indices = list(range(n_train)) + list(all_centers)
        unlabeled_indices = np.where(all_labels == -1)[0]
        x_all = np.vstack([x_train, x_test])
        x_all_torch = torch.from_numpy(x_all).float()
        unique_labels = np.unique(y_train)
        n_ways = len(unique_labels)
        if self.use_pslp:
            P = self.build_propagation_matrix(x_all, distances, k=self.k_graph, alpha=self.alpha_pslp)
        else:
            P = None
        mus = torch.zeros(n_ways, x_all.shape[1])
        for i, class_label in enumerate(unique_labels):
            support_indices = np.where(y_train == class_label)[0]
            center_indices = class_center_map.get(class_label, [])
            class_reference_indices = list(support_indices) + center_indices
            if len(class_reference_indices) > 0:
                class_samples = x_all[class_reference_indices]
                if hasattr(self, 'current_rho'):
                    sample_densities = self.current_rho[class_reference_indices]
                    weights = torch.from_numpy(sample_densities).float()
                    weights = weights / weights.sum()
                    class_samples_torch = torch.from_numpy(class_samples).float()
                    mus[i] = class_samples_torch.T @ weights
                else:
                    mus[i] = torch.from_numpy(class_samples).float().mean(0)
        support_labels_idx = torch.tensor(
            [np.where(unique_labels == y_train[i])[0][0] for i in range(n_train)],
            dtype=torch.long,
        )
        soft_km_probas_for_auc = None
        for _ in range(self.nIter_km):
            dist2 = torch.cdist(x_all_torch, mus).pow(2)
            probas = F.softmax(-dist2 * self.T_km, dim=1)
            soft_km_probas_for_auc = probas.detach().clone()
            Z = probas.clone()
            for idx in labeled_indices:
                label_idx = np.where(unique_labels == all_labels[idx])[0][0]
                Z[idx] = 0
                Z[idx, label_idx] = 1.0
            if self.use_pslp and P is not None:
                Z_propagated = torch.matmul(P, Z)
                if self.balanced == 'balanced':
                    Z_propagated = self.compute_optimal_transport(
                        M=Z_propagated,
                        n_support=n_train,
                        support_labels=support_labels_idx,
                        epsilon=self.ot_epsilon,
                        max_iters=self.ot_max_iters,
                    )
                else:
                    row_sum = Z_propagated.sum(dim=1, keepdim=True)
                    Z_propagated = Z_propagated / (row_sum + 1e-08)
                probas = Z_propagated
            elif self.balanced == 'balanced':
                probas = self.compute_optimal_transport(
                    M=Z,
                    n_support=n_train,
                    support_labels=support_labels_idx,
                    epsilon=self.ot_epsilon,
                    max_iters=self.ot_max_iters,
                )
            else:
                probas = Z
            probas_sum = probas.sum(dim=0, keepdim=True).T
            mus_new = probas.T @ x_all_torch / (probas_sum + 1e-08)
            if self.use_pslp:
                mus = (1 - self.beta_pslp) * mus + self.beta_pslp * mus_new
            else:
                mus = mus_new
        final_probas = probas.numpy()
        for i in unlabeled_indices:
            assigned_class_idx = final_probas[i].argmax()
            assigned_label = unique_labels[assigned_class_idx]
            all_labels[i] = assigned_label
        if soft_km_probas_for_auc is None:
            soft_km_probas_for_auc = probas.detach().clone()
        return (all_labels, mus, probas, soft_km_probas_for_auc)

    def run_task(self, data, label, args):
        """Evaluate a batch of few-shot tasks and return per-task metrics."""
        acc_list = []
        f1_list = []
        auc_list = []

        # Preprocess all tasks once before task-wise adaptation.
        data = preprocess(data, self.beta)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_tasks = data.shape[0]
        progress_interval = max(1, n_tasks // 10)
        print(f'[DPGS] {self.balanced} evaluation: {n_tasks} tasks, {args.n_ways}-way {args.shot}-shot')
        for i in range(data.shape[0]):
            data_i = data[i].to(self.device)
            label_i = label[i].to(self.device)
            n_lsample = args.n_ways * args.shot
            x_train = data_i[:n_lsample].cpu().numpy()
            y_train = label_i[:n_lsample].cpu().numpy()
            x_test = data_i[n_lsample:].cpu().numpy()
            y_test = label_i[n_lsample:].cpu().numpy()
            n_train = len(x_train)
            x_all = np.vstack([x_train, x_test])

            # Step 1: prototype-guided feature transformation.
            x_all = prototype_contrast_transform(
                x_all,
                x_train,
                y_train,
                gamma=self.gamma,
                use_adaptive_weights=True,
                distance_metric='euclidean',
            )
            x_train = x_all[:n_train]
            x_test = x_all[n_train:]
            feature_dim = x_all.shape[1]
            self.initialize_film_modules(feature_dim)
            x_all_torch = torch.from_numpy(x_all).float().to(self.device)

            # Step 2: learn a task-specific FiLM transformation.
            rho_init, distances_init, nearest_higher_density_init = self.calculate_density_and_distance(x_all)
            init_centers_set, init_class_center_map, init_labels = (
                self.assign_chain_labels_and_find_centers(
                    x_train,
                    y_train,
                    x_test,
                    rho_init,
                    nearest_higher_density_init,
                    distances_init,
                )
            )
            init_centers = list(init_centers_set)
            init_labels, mus_init, probas_init, _ = self.assign_unlabeled_points(
                x_train,
                y_train,
                x_test,
                init_labels,
                distances_init,
                n_train,
                init_centers,
                init_class_center_map,
            )
            labeled_indices = list(range(n_train)) + init_centers
            labeled_labels = init_labels[labeled_indices]
            unique_labels = np.unique(y_train)
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            mus_init = mus_init.to(self.device)
            probas_init = probas_init.to(self.device)
            cond = torch.cat([mus_init.mean(dim=0), mus_init.std(dim=0)]).unsqueeze(0)
            labels_l = torch.tensor(
                [label_to_idx[l] for l in labeled_labels],
                device=self.device,
                dtype=torch.long,
            )
            y_train_mapped = torch.tensor([label_to_idx[l] for l in y_train], device=self.device, dtype=torch.long)
            best_loss = float('inf')
            best_gamma = None
            best_beta = None
            for _ in range(self.film_inner_steps):
                self.transform_opt.zero_grad()
                gamma, beta = self.cond_mlp(cond)
                gamma = gamma.squeeze(0)
                beta = beta.squeeze(0)
                x_transformed = gamma.unsqueeze(0) * x_all_torch + beta.unsqueeze(0)
                x_transformed = F.normalize(x_transformed, p=2, dim=1)
                probas_for_mus = probas_init.detach()
                mus_fixed = []
                for cls_idx in range(len(unique_labels)):
                    weights = probas_for_mus[:, cls_idx]
                    weights_sum = weights.sum()
                    if weights_sum > 1e-08:
                        m = (weights.unsqueeze(1) * x_transformed).sum(0) / weights_sum
                        m = F.normalize(m.unsqueeze(0), p=2, dim=1).squeeze(0)
                    else:
                        m = torch.zeros(feature_dim, device=self.device)
                    mus_fixed.append(m)
                mus_fixed = torch.stack(mus_fixed, dim=0)
                loss_ce = self.compute_prototype_ce_loss(x_transformed[:n_train], y_train_mapped, mus_fixed)
                x_query_transformed = x_transformed[n_train:]
                rho_query_torch = torch.from_numpy(rho_init[n_train:]).float().to(self.device)
                probas_query = probas_init[n_train:]
                prob_max_query, pseudo_labels_query_idx = probas_query.max(dim=1)
                loss_cluster, _ = self.compute_cluster_structure_loss(
                    x_query=x_query_transformed,
                    y_pseudo=pseudo_labels_query_idx,
                    prob_max=prob_max_query,
                    rho_query=rho_query_torch,
                    mus=mus_fixed,
                    prob_thresh=self.cluster_prob_thresh,
                    rho_thresh_percentile=self.cluster_rho_thresh_percentile,
                )
                x_labeled = x_transformed[labeled_indices]
                loss_intra_compact = self.compute_intra_class_compactness_loss(x_labeled, labels_l, mus_fixed)
                loss_inter_sep = self.compute_inter_class_separation_loss(mus_fixed)
                loss_total = (
                    self.lambda_ce * loss_ce
                    + self.lambda_cluster * loss_cluster
                    + self.lambda_intra_compact * loss_intra_compact
                    + self.lambda_inter_sep * loss_inter_sep
                )
                loss_total.backward()
                self.transform_opt.step()
                if loss_total.item() < best_loss:
                    best_loss = loss_total.item()
                    best_gamma = gamma.detach().clone()
                    best_beta = beta.detach().clone()
            with torch.no_grad():
                x_all_transformed_final = best_gamma.unsqueeze(0) * x_all_torch + best_beta.unsqueeze(0)
                x_all_transformed_final = F.normalize(x_all_transformed_final, p=2, dim=1)
                x_all = x_all_transformed_final.cpu().numpy()
                x_train = x_all[:n_train]
                x_test = x_all[n_train:]

            # Step 3: rebuild density chains and infer query labels.
            rho, distances, nearest_higher_density = self.calculate_density_and_distance(x_all)
            self.current_rho = rho
            final_centers_set, final_class_center_map, final_labels = (
                self.assign_chain_labels_and_find_centers(
                    x_train,
                    y_train,
                    x_test,
                    rho,
                    nearest_higher_density,
                    distances,
                )
            )
            final_centers = list(final_centers_set)
            final_labels, _, _, final_auc_probas = self.assign_unlabeled_points(
                x_train,
                y_train,
                x_test,
                final_labels,
                distances,
                n_train,
                final_centers,
                final_class_center_map,
            )
            query_predictions = final_labels[n_train:]
            acc = accuracy_score(y_test, query_predictions)
            acc_list.append(acc)
            task_classes = np.unique(y_train)
            f1 = f1_score(y_test, query_predictions, average='macro', labels=task_classes, zero_division=0)
            f1_list.append(f1)
            query_scores = final_auc_probas[n_train:].detach().cpu().numpy()
            query_scores = query_scores / (query_scores.sum(axis=1, keepdims=True) + 1e-12)
            try:
                if len(task_classes) == 2:
                    auc = roc_auc_score(y_test, query_scores[:, 1])
                else:
                    auc = roc_auc_score(y_test, query_scores, labels=task_classes, multi_class='ovr', average='macro')
            except ValueError:
                auc = np.nan
            auc_list.append(auc)
            completed = i + 1
            if completed % progress_interval == 0 or completed == n_tasks:
                print(f'[DPGS] progress {completed}/{n_tasks}')
        acc_array = np.array(acc_list)
        acc_mean = acc_array.mean()
        acc_conf = 1.96 * acc_array.std(ddof=1) / np.sqrt(len(acc_array))
        f1_array = np.array(f1_list)
        auc_array = np.array(auc_list, dtype=float)
        print(f'[DPGS] mean accuracy: {acc_mean:.4f} +/- {acc_conf:.4f}')
        return {
            'acc': acc_array.reshape(-1, 1),
            'f1': f1_array.reshape(-1, 1),
            'auc': auc_array.reshape(-1, 1),
        }

def scaleEachUnitaryDatas(datas):
    """L2-normalize every feature vector."""
    norms = datas.norm(dim=2, keepdim=True)
    return datas / norms

def QRreduction(datas):
    """Apply task-wise QR reduction."""
    ndatas = torch.linalg.qr(datas.permute(0, 2, 1), 'reduced').R
    ndatas = ndatas.permute(0, 2, 1)
    return ndatas

def preprocess(data, beta):
    """Apply power transform, QR reduction, and L2 normalization."""
    data = torch.pow(data + 1e-06, beta)
    data = QRreduction(data)
    data = scaleEachUnitaryDatas(data)
    return data
