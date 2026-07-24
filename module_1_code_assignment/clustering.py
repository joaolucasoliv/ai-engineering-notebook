# lastly we are going to create a beginner-friendly clustering model.
# we will use K-Means to discover different road-maitenance profiles
# a clustering model discovers categories by itself, based on similarities in the data
# K-Means is a type of unsupervised learning algorithm used to group unlabelled data into disctinct clusters.
    # the algorithm repeatedly assigns points to the centroid and recalculates the center's average.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from module_1_code_assignment.regression import crack_severity

# random_state ensure that we get the same random results
# every time we run the program.
RANDOM_STATE = 42

# create a random-number generator
rng = np.random.default_rng(RANDOM_STATE) # rng stands for random number generator

# ======================
# step 1. CREATE A ****SYNTHETIC**** DATASET
# ======================

roads_per_cluster = 100

# Roads with generally lower damage
roads_low_damage = pd.DataFrame (
    {
        "potholes": rng.integers(
            low=0,
            high=8,
            size=roads_per_cluster,
        ),
        "crack_severity": rng.uniform(
            low=0,
            high=3.5,
            size=roads_per_cluster,
        ),
        "daily_traffic": rng.integers(
            low=500,
            high=6_000,
            size=roads_per_cluster,
        ),
    }
)

# roads with moderate damage
roads_moderate_damage = pd.DataFrame(
    {
        "potholes": rng.integers(
            low=6,
            high=19,
            size=roads_per_cluster,
        ),
        "crack_severity": rng.uniform(
            low=3,
            high=7.0,
            size=roads_per_cluster,
        ),
        "daily_traffic": rng.integers(
            low=4_000,
            high=17_000,
            size=roads_per_cluster,
        ),
    }
)

# roads with high damage
roads_high_damage = pd.DataFrame(
    {
        "potholes": rng.integers(
            low=15,
            high=31,
            size=roads_per_cluster,
        ),
        "crack_severity": rng.uniform(
            low=6.5,
            high=10,
            size=roads_per_cluster,
        ),
        "daily_traffic": rng.integers(
            low=13_000,
            high=35_000,
            size=roads_per_cluster,
        ),
    }
)

# combine all roads into one unique dataset
roads = pd.concat(
[roads_low_damage, roads_moderate_damage, roads_high_damage,],
    ignore_index=True,              # ignore_index=True is used to avoid the index being reset to a default value
)

# shuffle the rows so they are not organized by CONDITION
roads = roads.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

print("First five roads:")
print(roads.head())

# ======================
# step 2. SELECT THE FEATURES
# ======================

features = roads[ #check
    [
        "potholes",
        "crack_severity",
        "daily_traffic",
    ]
]

# there is no target variable: no y_cost or y_urgent since K-MEANS ONLY RECEIVES THE ROAD CHARACTERISTICS

# ======================
# step 3. STANDARDIZE THE FEATURES
# ======================

scaler = StandardScaler()                           # StandardScaler() scales features to have a mean of 0 and a standard deviation of 1.
features_scaled = scaler.fit_transform(features)

# StandardScaler prevents daily_taffic from dominating the model. Without scaling, traffic would have a much larger numerical influence.

# ======================
# step 4. CREATE THE K-MEANS MODEL
# ======================

model = KMeans(
    n_clusters=3,   # means that we are asking the model to sepaate the roads into three groups
    random_state=RANDOM_STATE,
    n_init=10,      # means that K-Means will try ten different starting positions and keep the best result
)

# ======================
# step 5. TRAIN THE MODEL AND CREATE THE CLUSTERS
# ======================

roads["cluster"] = model.fit_predict(features_scaled)

print("\nCluster Assigned to each road:")
print(roads.head())

#the cluster numbers may be: 0, 1, 2; these numbers do not automatically mean:
    # 0 = good; 1 = moderate; 2 = bad.
    # Cluster IDs are arbitrary


# ======================
# step 6. UNDERSTAND THE CLUSTER CENTERS
# ======================

# the original cluster centers are standardized
# inverse_transform converts them back to real units.

original_centers = scaler.inverse_transform(model.cluster_centers_)         # model.cluster_centers_ is a numpy array containing the cluster centers

cluster_centers = pd.DataFrame(
    original_centers,
    columns=features.columns,
)

cluster_centers["cluster"] = cluster_centers.index

print("\nCluster centers:")
print(cluster_centers.round(2)) # round to two decimal places

# each center represents the average road profile discoverd inside that cluster.


# ======================
# step 7. HUMAN-READABLE NAMES TO THE CLUSTERS
# ======================

cluster_order = np.argsort(                     # np.argsort() returns the indices that would sort the array.
    model.cluster_centers_.mean(axis=1)         # calculate horizontally across the columns.
)

profile_names = [
    "Lower-maintenance",
    "Moderate-maintenance",
    "High-maintenance"
]

cluster_name_map = {        # this dictionary maps the cluster numbers to human-readable names
    cluster_id: profile_names[position]
    for position, cluster_id in enumerate(cluster_order)           # this line of code iterates through the cluster numbers and assigns them to human-readable names. 0 = Lower-maintenance, 1 = Moderate-maintenance, 2 = High-maintenance
}

roads["road_profile"] = roads["cluster"].map(
    cluster_name_map            # this line maps the cluster numbers to human-readable names
)

cluster_centers["road_profile"] = cluster_centers[
    "cluster"
].map(cluster_name_map)             # this line maps the cluster numbers to human-readable names. .map is a pandas function that applies a function to each element in a column.

print("\nInterpreted cluster centers:")
print(
    cluster_centers[
        [
        "cluster",
        "road_profile",
        "potholes",
        "crack_severity",
        "daily_traffic"
        ]
    ].round(2)
)



# ======================
# step 8. EVALUATE THE CLUSTERING
# ======================

silhouette = silhouette_score(
    features_scaled,
    roads["cluster"],
)

print(f"\nSilhouette score: {silhouette:.3f}")


# the silhouette score is a measure of how well the model is able to separate the clusters.
# score generally ranges from -1 to 1. Close to 1, the clusters are clearly separated.
    # close to 0, the clusters are overlapping.
    # below 0, some roads may have been assigned to the wrong cluster.


# ======================
# step 9. VISUALIZE THE CLUSTERS
# ======================

for cluster_id in cluster_order:            # this loop iterates through the cluster numbers
    cluster_roads = roads[
        roads["cluster"] == cluster_id      # this line selects the roads that belong to the current cluster
    ]
    plt.scatter(
        cluster_roads["potholes"],
        cluster_roads["crack_severity"],
        label=cluster_name_map[cluster_id],
        alpha=0.6,
    )

# add the center of each cluster to the graph
plt.scatter(
    cluster_centers["potholes"],
    cluster_centers["crack_severity"],
    marker="X",                         # marker style
    s=250,                              # size of the marker
    label="Cluster Centers",)


plt.xlabel("Number of potholes")
plt.ylabel("Crack Severity")
plt.title("Clustering of Road Characteristics with K-Means")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ======================
# step 10. RECEIVE DATA FOR A NEW ROAD
# ======================

print("\nEnter information about a new road:")

new_potholes = float(
    input("Number of potholes: ")
)

new_crack_severity = float(
    input("Crack severity from 0 to 10:")
)

new_daily_traffic = float(
    input("Expected daily traffic per day:")
)

new_road = pd.DataFrame(
    {
        "potholes": [new_potholes],
        "crack_severity": [new_crack_severity],
        "daily_traffic": [new_daily_traffic],
    }
)

# Apply the same scaling used during training
scaled_new_road = scaler.transform(new_road)

# identify the nearest cluster
new_road_cluster = model.predict(
    scaled_new_road
)[0]            # [0] is the first cluster


new_road_profile = cluster_name_map[
    new_road_cluster
]

print("Clustering result:")
print(f"Cluster: {new_road_cluster}")
print(f"Road Profile: {new_road_profile}")
