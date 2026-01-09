def create_clusters(self, data_loader):
    all_data = []
    print("Clustering observations...")
    # Iterate over the batches
    for batch in tqdm(data_loader, desc="Processing Batches", unit="batch"):
        batch_size = batch["action"].shape[0]

        # Extract the relevant information
        nobs = self.normalizer.normalize(batch["obs"])
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        this_nobs = dict_apply(
            nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        proprio = []
        for key in self.lowdim_key:  # ['point_cloud', 'agent_pos']
            proprio.append(this_nobs[key].reshape(batch_size, -1))
        proprio = torch.cat(proprio, dim=-1)
        all_data.append(proprio)

    # Concatenate all processed batches
    all_data = torch.cat(all_data, dim=0)  # 900,3120

    result, num_clust, _ = FINCH(all_data.cpu().numpy())  #
    labels = result[:, -1]
    print("cluster number is:", num_clust)

    B, D = all_data.shape
    N = num_clust[-1]
    # Initialize centers array
    centers = torch.zeros((N, D), device=self.device)
    # Iterate over each class
    for class_idx in range(N):
        # Find samples belonging to the current class
        class_samples = all_data[labels == class_idx]
        print("class_idx", class_idx, "class_samples.shape:", class_samples.shape)
        if len(class_samples) > 0:
            # Compute the mean of the samples as the center
            centers[class_idx] = torch.mean(class_samples, dim=0)
        else:
            # If no samples belong to this class, leave the center as zeros
            centers[class_idx] = torch.zeros(D)
    self.centroids = centers

    # #Perform k-means clustering
    # self.centroids, _ = self.kmeans(all_data,num_clust[-1] )#64

    return self.centroids  #:cluster*3120


def assign_batch_to_clusters(self, vectors, centroids):
    distances = torch.cdist(vectors, centroids, p=2)  # Shape: (batch_size, num_clusters)

    # Find the closest cluster for each vector
    cluster_indices = torch.argmin(distances, dim=1)  # Shape: (batch_size,)
    # min_distances = torch.min(distances, dim=1).values  # Shape: (batch_size,)

    return cluster_indices


def kmeans(self, X, num_clusters, num_iterations=1000, tolerance=1e-7):
    num_samples, num_features = X.shape
    num_clusters = min(num_clusters, num_samples)

    # K-means++ initialization
    centroids = torch.zeros(num_clusters, num_features).to(X.device)
    # Choose the first centroid randomly
    centroids[0] = X[torch.randint(0, num_samples, (1,))]

    # Compute the remaining centroids
    for i in range(1, num_clusters):
        # Compute the distance from each point to the closest centroid
        distances = torch.cdist(X, centroids[:i], p=2).min(dim=1)[0]  # Shape: (num_samples,)

        # Select the next centroid with probability proportional to distance squared
        probs = distances ** 2
        probs /= probs.sum()  # Normalize to form a probability distribution
        next_centroid_idx = torch.multinomial(
            probs, 1
        )  # Choose one index based on the probability distribution
        centroids[i] = X[next_centroid_idx]

    for i in range(num_iterations):
        # Compute distances between samples and centroids
        distances = torch.cdist(
            X, centroids, p=2
        )  # Shape: (num_samples, num_clusters)
        labels = torch.argmin(
            distances, dim=1
        )  # Assign labels based on closest centroid

        # Update centroids
        new_centroids = torch.stack(
            [X[labels == k].mean(dim=0) for k in range(num_clusters)]
        )

        # Handle empty clusters by reinitializing them randomly
        for k in range(num_clusters):
            if torch.isnan(new_centroids[k]).any():
                new_centroids[k] = X[torch.randint(0, num_samples, (1,))]

        # Check for convergence
        if torch.norm(new_centroids - centroids, p="fro") < tolerance:
            break

        centroids = new_centroids

    return centroids, labels