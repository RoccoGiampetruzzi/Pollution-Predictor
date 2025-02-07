import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_with_different_angles(data, color_array, label_vector):
    """ 
    Function used to plot 3 dimensional data seen from different angles
    """
    
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    fig = plt.figure(figsize=(12, 4))
    colors = [color_array[label] for label in label_vector]
    
    angles = [(20, 30), (45, 45), (60, 120)]

    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.scatter(x, y, z, c=colors, marker='o')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev}, azim={azim}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    plt.show()
    
def compute_class_balance(df, class_column):
    classes = df[class_column].value_counts().to_dict()
    return classes


def augment_dataframe_with_convexcomb(df, class_column):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    df_augmented = df.copy()
    classes = df[class_column].value_counts()

    minority_class = 0 if classes[0] < classes[1] else 1
    n_samples = int(abs(classes[0] - classes[1]))

    df_minority_np = df[df[class_column] == minority_class].to_numpy()

    row_indices = np.random.randint(0, len(df_minority_np), size=(n_samples, 1))
    row_indices_2 = np.zeros((n_samples, 1), dtype=int)

    for i, index in enumerate(row_indices):
        
        temp_index = np.where(
                        (df_minority_np[:, 0] == df_minority_np[index,0]) &  # Column 1: 'season'
                        (df_minority_np[:, 7] == df_minority_np[index,7]) &  # Column 2: 'cbwd_NE'
                        (df_minority_np[:, 8] == df_minority_np[index,8]) &  # Column 3: 'cbwd_NW'
                        (df_minority_np[:, 9] == df_minority_np[index,9])    # Column 4: 'cbwd_SE'
                    )[0]

        while len(temp_index) < 2:

            index = np.random.randint(0, len(df_minority_np), size=(1, 1))
            temp_index = np.where(
                            (df_minority_np[:, 0] == df_minority_np[index,0]) &  # Column 1: 'season'
                            (df_minority_np[:, 7] == df_minority_np[index,7]) &  # Column 2: 'cbwd_NE'
                            (df_minority_np[:, 8] == df_minority_np[index,8]) &  # Column 3: 'cbwd_NW'
                            (df_minority_np[:, 9] == df_minority_np[index,9])    # Column 4: 'cbwd_SE'
                        )[0]
        
        row_indices[i] = index
        row_indices_2[i] = np.random.choice(temp_index)        

    lambdas = np.random.rand(n_samples, 1)
    row1 = df_minority_np[row_indices[:, 0]]
    row2 = df_minority_np[row_indices_2[:, 0]]
    new_datapoints = lambdas * row1 + (1 - lambdas) * row2
    new_datapoints[:, -1] = minority_class  

    df_newdatapoint = pd.DataFrame(new_datapoints, columns=df.columns)
    df_augmented = pd.concat([df_augmented, df_newdatapoint], ignore_index=True)

    return df_augmented

def augment_dataframe_with_noise(df, label_column, noise_scale=0.01, random_state=42):

    np.random.seed(random_state)
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    classes = df[label_column].value_counts()

    minority_class = 0 if classes[0] < classes[1] else 1
    n_samples = int(abs(classes[0] - classes[1]))

    feature_stds = df[df.columns[:10]].std().to_numpy()
    df_minority_np = df[df[label_column] == minority_class].to_numpy()[:,:10]
    
    samples = np.random.randint(0, len(df_minority_np), size=(n_samples))
    noise_vector = np.random.normal(loc=0, scale=noise_scale * feature_stds, size=(n_samples, len(feature_stds)))

    new_generated_samples = df_minority_np[samples] + noise_vector
    
    new_samples_df = pd.DataFrame(new_generated_samples, columns=df.columns[:10])
    new_samples_df[label_column] = minority_class
    
    df_augmented = pd.concat([df, new_samples_df], ignore_index=True)
    
    return df_augmented
