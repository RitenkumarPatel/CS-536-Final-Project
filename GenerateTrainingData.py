import numpy as np
rng = np.random.default_rng()

class GenerateTrainingData:

    def __init__(self, num_features: int, STD_DEV: float):
        self.num_features = num_features
        self.STD_DEV = STD_DEV

        if self.num_features < 101:
            raise ValueError("num_features has to be greater than 100")
    
    def generate_latents_and_targets(self, num_data_points: int):
        # A1 ... An are latents, B1 ... Bn are targets
        X_mat = np.zeros((num_data_points,self.num_features))
        Y_vec = np.zeros(num_data_points)
        for j in range(num_data_points):
            X = np.zeros(self.num_features)
            X[0] = 1
            for idx in range(1, 51):
                X[idx] = rng.normal()

            for idx in range(51, 61):
                X[idx] = X[1] + 0.5 * X[idx-50] + rng.normal(0,self.STD_DEV)

            for idx in range(61, 71):
                X[idx] = X[idx-60] - X[idx-50] + X[idx-40] + rng.normal(0,self.STD_DEV)

            for idx in range(71, 81):
                X[idx] = X[6*(idx-70)] + 3 * X[idx-70] + rng.normal(0,self.STD_DEV)

            for idx in range(81, 91):
                X[idx] = 5 - X[idx-10]

            for idx in range(91, self.num_features):
                X[idx] = 0.5 * X[50+(idx-90)] + 0.5 * X[50+(idx-80)] + rng.normal(0, self.STD_DEV)

            Y = rng.normal(0, 0.1)
            for i in range(1,51):
                Y += (-0.88)**i * X[2 * i]
            
            X_mat[j] = X
            Y_vec[j] = Y
        
        return X_mat, Y_vec
    

    def generate_observations(self, X_mat: np.ndarray):
        indices = np.arange(1, self.num_features + 1)  # 1-indexed
        noise_scales = self.STD_DEV * np.log(indices)

        noise = rng.normal(
            loc=0,
            scale=noise_scales,
            size=(X_mat.shape[0], self.num_features)
        )

        C_mat = X_mat + noise
        return C_mat
    
    def generate_data(self, num_data_points: int):
        A_mat, B_vec = self.generate_latents_and_targets(num_data_points)
        C_mat = self.generate_observations(A_mat)
        return A_mat, B_vec, C_mat
