"""
Imports
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import time
import os
import hashlib
import umap

from sklearn.decomposition import PCA, FastICA
from openTSNE import TSNE
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap


"""
CliffWalkingVisualizer
"""

class CliffWalkingVisualizer:
    def __init__(self, grid_height=4, grid_width=12, data_files=None, cache_dir='cache'):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.total_states = grid_height * grid_width
        self.data_files = data_files or {
            'expected_sarsa': 'data/cliff_walking/expected_sarsa.npy',
            'q_learning': 'data/cliff_walking/q_learning.npy',
            'random_policy': 'data/cliff_walking/random.npy',
            'sarsa': 'data/cliff_walking/sarsa.npy'
        }
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)  # Ensure cache directory exists

        # Load and preprocess data
        self.algorithms_data = self.load_and_preprocess_data()

    def generate_cache_path(self, episodes_steps, algorithm_name, method, encoding_type, sample_n):
        # Convert each episode to a contiguous array, then stack them
        contiguous_episodes = [np.ascontiguousarray(episode) for episode in episodes_steps]
        episodes_steps_hash = hashlib.md5(np.vstack(contiguous_episodes)).hexdigest()
        key_str = f"{algorithm_name}_{encoding_type}_{method}_{sample_n}_{episodes_steps_hash}"
        cache_file = os.path.join(self.cache_dir, f"{key_str}.npy")
        return cache_file

    def load_and_preprocess_data(self):
        data = {}
        for name, file in self.data_files.items():
            with open(file, 'rb') as f:
                episodes_data = np.load(f, allow_pickle=True)
                df = self.create_dataframe(episodes_data, name)
                df = self.precompute_encodings(df)
                data[name] = df
        return data

    def create_dataframe(self, episodes_data, algorithm_name):
        columns = ['state', 'action', 'reward', 'next_state', 'done']
        all_episodes = []
        for episode_idx, episode in enumerate(episodes_data):
            episode_df = pd.DataFrame(episode, columns=columns)
            episode_df['episode'] = episode_idx
            all_episodes.append(episode_df)
        df = pd.concat(all_episodes, ignore_index=True)
        df['algorithm'] = algorithm_name
        return df

    def state_to_grid_position(self, state):
        return divmod(state, self.grid_width)

    def one_hot_encode_state(self, state):
        one_hot_vector = np.zeros(self.total_states)
        one_hot_vector[state] = 1
        return one_hot_vector

    def precompute_encodings(self, df):
        # One-hot encoding
        one_hot_encoded = np.zeros((len(df), self.total_states))
        for idx, state in enumerate(df['state']):
            one_hot_encoded[idx] = self.one_hot_encode_state(state)
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=[f'state_{i}' for i in range(self.total_states)])

        # Manhattan encoding
        manhattan_encoded = np.zeros((len(df), self.total_states))
        for idx, state in enumerate(df['state']):
            for target_state in range(self.total_states):
                manhattan_encoded[idx, target_state] = self.manhattan_distance(state, target_state)
        manhattan_df = pd.DataFrame(manhattan_encoded, columns=[f'manhattan_state_{i}' for i in range(self.total_states)])

        # Concatenate the encodings with the original DataFrame
        df = pd.concat([df.reset_index(drop=True), one_hot_df, manhattan_df], axis=1)
        return df

    def manhattan_distance(self, state1, state2):
        pos1 = self.state_to_grid_position(state1)
        pos2 = self.state_to_grid_position(state2)
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def sample_episodes(self, df, n=10):
        unique_episodes = df['episode'].unique()
        sampled_episodes = unique_episodes[::n]
        return df[df['episode'].isin(sampled_episodes)]

    def extract_encoded_states(self, df, encoding_type='one-hot'):
        if encoding_type == 'one-hot':
            return df.filter(like='state_').values
        elif encoding_type == 'manhattan':
            return df.filter(like='manhattan_state_').values
        else:
            raise ValueError("Invalid encoding type. Use 'one-hot' or 'manhattan'.")

    def extract_steps_per_episode(self, df, encoding_type='one-hot'):
        episodes_steps = []
        episodes_states = []  # New list to track original states
        unique_episodes = df['episode'].unique()
        for episode in unique_episodes:
            episode_df = df[df['episode'] == episode]
            encoded_states = self.extract_encoded_states(episode_df, encoding_type)
            original_states = episode_df['state'].values  # Store original states
            episodes_steps.append(encoded_states)
            episodes_states.append(original_states)
        return episodes_steps, episodes_states

    def down_project_episodes(self, episodes_steps, episodes_states, n_components=2, algorithm_name='expected_sarsa', method='PCA', encoding_type='one-hot', sample_n=10):
        # Generate a unique cache path
        cache_path = self.generate_cache_path(episodes_steps, algorithm_name, method, encoding_type, sample_n)

        # Check if the cache file exists
        if os.path.exists(cache_path):
            print("Loading down-projected data from cache.")
            down_projected = np.load(cache_path, allow_pickle=True)
        else:
            print("Computing down-projection.")
            all_steps = np.vstack(episodes_steps)

            # Select projection method
            if method == 'PCA':
                projector = PCA(n_components=n_components)
                down_projected = projector.fit_transform(all_steps)
            elif method == 't-SNE':
                projector = TSNE(n_components=n_components, perplexity=30, random_state=42)
                down_projected = projector.fit(all_steps)
            elif method == 'UMAP':
                projector = umap.UMAP(n_components=n_components, random_state=42)
                down_projected = projector.fit_transform(all_steps)
            elif method == 'ICA':
                projector = FastICA(n_components=n_components, random_state=42)
                down_projected = projector.fit_transform(all_steps)
            else:
                raise ValueError("Invalid method. Use 'PCA', 't-SNE', 'UMAP', or 'ICA'.")

            # Save the down-projected data to the cache
            np.save(cache_path, down_projected)

        # Split down_projected back into episodes
        down_projected_episodes = []
        down_projected_states = []  # Track states for each episode
        start = 0
        for episode_steps, episode_states in zip(episodes_steps, episodes_states):
            end = start + len(episode_steps)
            down_projected_episodes.append(down_projected[start:end])
            down_projected_states.append(episode_states)
            start = end

        return down_projected_episodes, down_projected_states

    def identify_state_clusters(self, down_projected_episodes, down_projected_states):
        all_points = np.vstack(down_projected_episodes)
        all_states = np.concatenate(down_projected_states)

        # Use DBSCAN to identify clusters
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(all_points)

        # Create a mapping of points to their original states
        state_points_map = {}
        for point, state in zip(all_points, all_states):
            if state not in state_points_map:
                state_points_map[state] = []
            state_points_map[state].append(point)

        # Calculate cluster centers and their corresponding states
        cluster_info = []

        # Process each unique state
        for state, points in state_points_map.items():
            points = np.array(points)
            if len(points) < 5:  # Skip states with very few occurrences
                continue

            # Calculate the center of all points for this state
            center = points.mean(axis=0)

            # Calculate density (number of points for this state)
            density = len(points)

            # Calculate text size based on density
            min_size = 8  # Increased minimum text size
            max_size = 24  # Increased maximum text size
            min_density = 5
            max_density = max(100, density)  # Adapt max density to the actual maximum

            normalized_density = (density - min_density) / (max_density - min_density)
            normalized_density = max(0, min(1, normalized_density))
            text_size = min_size + (max_size - min_size) * np.sqrt(normalized_density)

            cluster_info.append({
                'x': float(center[0]),
                'y': float(center[1]),
                'state': str(int(state)),
                'density': density,
                'text_size': float(text_size)
            })

        return pd.DataFrame(cluster_info)

    def calculate_density(self, down_projected_episodes):
        all_points = np.vstack(down_projected_episodes)
        unique_points, counts = np.unique(all_points, axis=0, return_counts=True)

        density_df = pd.DataFrame(unique_points, columns=['Axis 1', 'Axis 2'])
        density_df['Density'] = counts
        return density_df

    def create_connected_density_plot(self, down_projected_episodes, down_projected_states, algorithm_name):
        # Get cluster information
        cluster_df = self.identify_state_clusters(down_projected_episodes, down_projected_states)

        # Add color coding for start and goal states
        cluster_df['state_type'] = 'regular'
        cluster_df.loc[cluster_df['state'] == '36', 'state_type'] = 'start'
        cluster_df.loc[cluster_df['state'] == '47', 'state_type'] = 'goal'

        # Combine trajectory data
        data = []
        for episode_id, episode_data in enumerate(down_projected_episodes):
            for step_id, step in enumerate(episode_data):
                data.append({
                    'Axis 1': step[0],
                    'Axis 2': step[1],
                    'Episode': episode_id,
                    'Step': step_id
                })
        trajectory_df = pd.DataFrame(data)

        # Create density circles with color coding
        density_plot = alt.Chart(cluster_df).mark_circle(
            opacity=0.3
        ).encode(
            x='x:Q',
            y='y:Q',
            size=alt.Size(
                'density:Q',
                scale=alt.Scale(range=[100, 2000]),
                legend=None
            ),
            color=alt.Color(
                'state_type:N',
                scale=alt.Scale(
                    domain=['regular', 'start', 'goal'],
                    range=['blue', 'green', 'purple']
                ),
                legend=None
            ),
            tooltip=['state:N', 'density:Q', 'state_type:N']
        )

        # State labels with color-coded text
        labels = alt.Chart(cluster_df).mark_text(
            align='center',
            baseline='middle',
            fontWeight='bold'
        ).encode(
            x='x:Q',
            y='y:Q',
            text='state:N',
            size=alt.Size('text_size:Q', scale=None),
            color=alt.Color(
                'state_type:N',
                scale=alt.Scale(
                    domain=['regular', 'start', 'goal'],
                    range=['white', 'white', 'white']
                ),
                legend=None
            ),
            tooltip=['density:Q', 'state:N', 'state_type:N']
        )

        # Trajectory lines
        line_plot = alt.Chart(trajectory_df).mark_line(
            opacity=0.01,
            interpolate='natural'
        ).encode(
            x='Axis 1:Q',
            y='Axis 2:Q',
            detail='Episode:N',
            color=alt.value('blue')
        )

        # Combine all layers
        combined_plot = (line_plot + density_plot + labels).properties(
            title=f"State Transitions and Clusters - {algorithm_name}",
            width=700,
            height=700
        ).interactive()

        return combined_plot

    def process_and_visualize(self, algorithm_name, encoding_type='one-hot', down_project_method='PCA', sample_n=10, density_plot=False):
        # Retrieve the precomputed DataFrame
        df = self.algorithms_data[algorithm_name]
        sampled_df = self.sample_episodes(df, n=sample_n)
        steps, states = self.extract_steps_per_episode(sampled_df, encoding_type=encoding_type)

        # Down-project the episodes
        down_projected, down_projected_states = self.down_project_episodes(steps, states, n_components=2, algorithm_name=algorithm_name, method=down_project_method, sample_n=sample_n)

        if density_plot:
            return self.create_connected_density_plot(down_projected, down_projected_states, algorithm_name)
        else:
            # Create and return a standard line-based trajectory plot
            df_projected = self.create_downprojected_df(down_projected, algorithm_name)
            return self.visualize_algorithm(df_projected, algorithm_name)

    def visualize_manhattan_distances(self, agent_state):
        fig, ax = plt.subplots(figsize=(12, 4))
        manhattan_distances = np.zeros((self.grid_height, self.grid_width))

        for state in range(self.total_states):
            x, y = self.state_to_grid_position(state)
            manhattan_distances[x, y] = self.manhattan_distance(agent_state, state)

        cmap = plt.get_cmap('coolwarm')
        img = ax.imshow(manhattan_distances, cmap=cmap, extent=[0, self.grid_width, 0, self.grid_height])

        for state in range(self.total_states):
            x, y = self.state_to_grid_position(state)
            ax.text(y + 0.5, self.grid_height - x - 0.5, int(manhattan_distances[x, y]), ha='center', va='center', color='black')

        agent_x, agent_y = self.state_to_grid_position(agent_state)
        ax.scatter(agent_y + 0.5, self.grid_height - agent_x - 0.5, color='gray', s=200, label='Agent')

        plt.colorbar(img, ax=ax)
        ax.set_xticks(np.arange(self.grid_width))
        ax.set_yticks(np.arange(self.grid_height))
        ax.set_xticklabels(np.arange(self.grid_width))
        ax.set_yticklabels(np.arange(self.grid_height - 1, -1, -1))
        ax.grid(True)
        ax.set_title(f"Manhattan Distances from State {agent_state}")
        ax.legend(loc='upper left')
        plt.show()

    def visualize_grid_world(self, agent_state):
        fig, ax = plt.subplots(figsize=(12, 4))
        grid = np.zeros((self.grid_height, self.grid_width))

        start_state = 36
        goal_state = 47
        cliff_states = list(range(37, 47))

        for state in cliff_states:
            x, y = self.state_to_grid_position(state)
            grid[x, y] = 1  # Cliff

        start_x, start_y = self.state_to_grid_position(start_state)
        goal_x, goal_y = self.state_to_grid_position(goal_state)
        grid[start_x, start_y] = 2  # Start
        grid[goal_x, goal_y] = 3    # Goal

        colors = ['white', 'gray', 'blue', 'green']
        custom_cmap = ListedColormap(colors)
        ax.imshow(grid, cmap=custom_cmap, extent=[0, self.grid_width, 0, self.grid_height])

        agent_x, agent_y = self.state_to_grid_position(agent_state)
        ax.scatter(agent_y + 0.5, self.grid_height - agent_x - 0.5, color='purple', s=200, label='Agent')

        for state in range(self.total_states):
            x, y = self.state_to_grid_position(state)
            ax.text(y + 0.5, self.grid_height - x - 0.5, str(state), ha='center', va='center', color='black')

        ax.set_xticks(np.arange(self.grid_width))
        ax.set_yticks(np.arange(self.grid_height))
        ax.set_xticklabels(np.arange(self.grid_width))
        ax.set_yticklabels(np.arange(self.grid_height - 1, -1, -1))
        ax.grid(True)
        ax.set_title("Cliff Walking Grid World")
        ax.legend(loc='upper left')
        plt.show()