import tkinter as tk
from tkinter import filedialog, messagebox
import open3d as o3d
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import subprocess
import sys
import tempfile
import os


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x = y = 0
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief="solid", borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=4, ipady=2)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class PointCloudApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("open3d-aligner-gui")
        self.root.geometry("600x800")
        self.root.resizable(False, False)

        # --- Create main container frame to lock window size ---
        self.main_frame = tk.Frame(self.root, width=600, height=800)
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.pack_propagate(False)  # Prevent children from resizing root

        self.transformation = None
        self.source = None
        self.target = None
        self.source_sampled = None
        self.target_sampled = None

        # === File I/O Frame ===
        file_frame = tk.LabelFrame(self.main_frame, text="Load Point Cloud Files", padx=10, pady=10)
        file_frame.pack(fill="x", padx=10, pady=5)

        tk.Button(file_frame, text="Load Source (.asc)", command=self.load_source, width=20)\
            .grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.source_label = tk.Label(file_frame, text="No source file loaded", anchor="w")
        self.source_label.grid(row=0, column=1, padx=5, pady=2, sticky="w")

        tk.Button(file_frame, text="Load Target (.asc)", command=self.load_target, width=20)\
            .grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.target_label = tk.Label(file_frame, text="No target file loaded", anchor="w")
        self.target_label.grid(row=1, column=1, padx=5, pady=2, sticky="w")

        # === Preprocess Frame ===
        proprocess_frame = tk.LabelFrame(self.main_frame, text="Preprocess", padx=10, pady=10)
        proprocess_frame.pack(fill="x", padx=10, pady=5)

        # Remove Outliers button
        self.remove_outliers_btn = tk.Button(proprocess_frame, text="Remove Outliers", command=self.remove_outliers, width=20)
        self.remove_outliers_btn.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        # Down Sample label
        down_sample_label = tk.Label(proprocess_frame, text="Down Sample Rate:")
        down_sample_label.grid(row=0, column=1, sticky="e", padx=5, pady=2)
        ToolTip(down_sample_label, text="Down Sample Rate: ...")

        # Entry for down sample
        vcmd = (root.register(self.validate_float_0_to_1), "%P")
        self.sample_entry = tk.Entry(proprocess_frame, width=10, justify="right", validate="key", validatecommand=vcmd)
        self.sample_entry.insert(0, "0.3")  # Default value
        self.sample_entry.grid(row=0, column=2, sticky="w", padx=5, pady=2)

        # Down Sample button can stay next to entry 
        self.sample_btn = tk.Button(proprocess_frame, text="Down Sample", command=self.downsample, width=20)
        self.sample_btn.grid(row=0, column=3, sticky="w", padx=5, pady=2)

        # === Registration Parameters Frame ===
        param_frame = tk.LabelFrame(self.main_frame, text="Registration Parameters", padx=10, pady=10)
        param_frame.pack(fill="x", padx=10, pady=5)

        voxel_label = tk.Label(param_frame, text="Voxel Size:")
        voxel_label.grid(row=0, column=0, sticky="e", padx=5, pady=2)
        vcmd = (root.register(self.validate_non_negative_float), "%P")
        self.voxel_entry = tk.Entry(param_frame, width=10, justify="right", validate="key", validatecommand=vcmd)
        self.voxel_entry.insert(0, "1.0")  # Default value
        self.voxel_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        tk.Label(param_frame, text="").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ToolTip(voxel_label, text="Voxel Size: ...")

        radius_label = tk.Label(param_frame, text="Radius:")
        radius_label.grid(row=0, column=3, sticky="e", padx=5, pady=2)
        vcmd = (root.register(self.validate_non_negative_float), "%P")
        self.radius_entry = tk.Entry(param_frame, width=10, justify="right", validate="key", validatecommand=vcmd)
        self.radius_entry.insert(0, "4.0")  # Default value
        self.radius_entry.grid(row=0, column=4, sticky="w", padx=5, pady=2)
        tk.Label(param_frame, text="* Voxel Size").grid(row=0, column=5, sticky="w", padx=5, pady=2)
        ToolTip(radius_label, text="Radius: ...")

        max_nn_label = tk.Label(param_frame, text="Max_NN:")
        max_nn_label.grid(row=1, column=0, sticky="e", padx=5, pady=2)
        vcmd = (root.register(self.validate_non_negative_integer), "%P")
        self.max_nn_entry = tk.Entry(param_frame, width=10, justify="right", validate="key", validatecommand=vcmd)
        self.max_nn_entry.insert(0, "40")  # Default value
        self.max_nn_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        tk.Label(param_frame, text="").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        ToolTip(max_nn_label, text="Max_NN: ...")

        max_cor_distance_label = tk.Label(param_frame, text="Max Correspondence Distance:")
        max_cor_distance_label.grid(row=1, column=3, sticky="e", padx=5, pady=2)
        vcmd = (root.register(self.validate_non_negative_float), "%P")
        self.max_cor_distance_entry = tk.Entry(param_frame, width=10, justify="right", validate="key", validatecommand=vcmd)
        self.max_cor_distance_entry.insert(0, "1.4")  # Default value
        self.max_cor_distance_entry.grid(row=1, column=4, sticky="w", padx=5, pady=2)
        tk.Label(param_frame, text="* Voxel Size").grid(row=1, column=5, sticky="w", padx=5, pady=2)
        ToolTip(max_cor_distance_label, text="Max Correspondence Distance: ...")

        max_iter_label = tk.Label(param_frame, text="Max Iteration:")
        max_iter_label.grid(row=2, column=0, sticky="e", padx=5, pady=2)
        vcmd = (root.register(self.validate_non_negative_integer), "%P")
        self.max_iter_entry = tk.Entry(param_frame, width=10, justify="right", validate="key", validatecommand=vcmd)
        self.max_iter_entry.insert(0, "8000000")  # Default value
        self.max_iter_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        tk.Label(param_frame, text="").grid(row=2, column=2, sticky="w", padx=5, pady=2)
        ToolTip(max_iter_label, text="Max Iteration: ...")

        confidence_label = tk.Label(param_frame, text="Confidence:")
        confidence_label.grid(row=2, column=3, sticky="e", padx=5, pady=2)
        vcmd = (root.register(self.validate_float_0_to_1), "%P")
        self.confidence_entry = tk.Entry(param_frame, width=10, justify="right", validate="key", validatecommand=vcmd)
        self.confidence_entry.insert(0, "0.99999") # Default value
        self.confidence_entry.grid(row=2, column=4, sticky="w", padx=5, pady=2)
        tk.Label(param_frame, text="").grid(row=2, column=5, sticky="w", padx=5, pady=2)
        ToolTip(confidence_label, text="Confidence: ...")

        # === Registration + Output Frame ===
        reg_frame = tk.LabelFrame(self.main_frame, text="Registration Results", padx=10, pady=10)
        reg_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.register_btn = tk.Button(reg_frame, text="Register Point Clouds", command=self.register_point_clouds, width=22)
        self.register_btn.pack(pady=(0, 10))

        self.output_text = tk.Text(reg_frame, height=20, width=60)
        self.output_text.pack(fill="both", expand=True)

        # === View Frame ===
        view_frame = tk.LabelFrame(self.main_frame, text="Visualization", padx=10, pady=10)
        view_frame.pack(fill="x", padx=10, pady=5)

        # Configure grid columns to expand equally (3 columns now)
        view_frame.grid_columnconfigure(0, weight=1)
        view_frame.grid_columnconfigure(1, weight=1)
        view_frame.grid_columnconfigure(2, weight=1)

        btn_orig = tk.Button(view_frame, text="View Original Point Clouds", command=self.view_point_clouds, width=28)
        btn_orig.grid(row=0, column=0, padx=5, pady=0)

        btn_preprocessed = tk.Button(view_frame, text="View Preprocessed Point Clouds", command=self.view_preprocessed_point_clouds, width=28)
        btn_preprocessed.grid(row=0, column=1, padx=5, pady=0)

        btn_aligned = tk.Button(view_frame, text="View Aligned Point Clouds", command=self.view_aligned_point_clouds, width=28)
        btn_aligned.grid(row=0, column=2, padx=5, pady=0)
        

    def load_source(self):
        file_path = filedialog.askopenfilename(filetypes=[("ASC Files", "*.asc")])
        if file_path:
            self.source = o3d.io.read_point_cloud(file_path, format='xyz')
            relative_path = Path(file_path).relative_to(Path.cwd())
            self.source_label.config(text=str(relative_path))


    def load_target(self):
        file_path = filedialog.askopenfilename(filetypes=[("ASC Files", "*.asc")])
        if file_path:
            self.target = o3d.io.read_point_cloud(file_path, format='xyz')
            relative_path = Path(file_path).relative_to(Path.cwd())
            self.target_label.config(text=str(relative_path))


    def validate_non_negative_float(self, value_if_allowed):
        if value_if_allowed == "":
            return True 
        try:
            value = float(value_if_allowed)
            return value >= 0
        except ValueError:
            return False
        

    def validate_non_negative_integer(self, value_if_allowed):
        if value_if_allowed == "":
            return True 
        return value_if_allowed.isdigit()
        

    def validate_float_0_to_1(self, value_if_allowed):
        if value_if_allowed == "":
            return True 
        try:
            val = float(value_if_allowed)
            return 0 <= val <= 1
        except ValueError:
            return False


   # --- View Original Point Clouds ---
    def view_point_clouds(self):
        if self.source is None or self.target is None:
            messagebox.showerror("Error", "Load both point clouds first.")
            return

        # --- Save point clouds to temporary files ---
        source_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
        target_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
        o3d.io.write_point_cloud(source_file.name, self.source)
        o3d.io.write_point_cloud(target_file.name, self.target)

        source_file.close()
        target_file.close()

        # --- Launch Open3D window in separate process ---
        subprocess.Popen([sys.executable, "view_clouds_files.py", source_file.name, target_file.name])


    # --- View Preprocessed Point Clouds ---
    def view_preprocessed_point_clouds(self):
        if self.source is None or self.target is None:
            messagebox.showerror("Error", "Load both point clouds first.")
            return

        if self.source_sampled is None or self.target_sampled is None:
            messagebox.showerror("Error", "Run outlier removal and down sampling before viewing preprocessed point clouds.")
            return

        source_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
        target_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
        o3d.io.write_point_cloud(source_file.name, self.source_sampled)
        o3d.io.write_point_cloud(target_file.name, self.target_sampled)

        source_file.close()
        target_file.close()

        subprocess.Popen([sys.executable, "view_clouds_files.py", source_file.name, target_file.name])


    # --- View Aligned Point Clouds ---
    def view_aligned_point_clouds(self):
        if self.source is None or self.target is None:
            messagebox.showerror("Error", "Load both point clouds first.")
            return

        if self.transformation is None:
            messagebox.showerror("Error", "Run registration before viewing aligned point clouds.")
            return

        # aligned_source = copy.deepcopy(self.source) 
        aligned_source = copy.deepcopy(self.source_sampled) 
        aligned_source.transform(self.transformation)

        source_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
        target_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
        o3d.io.write_point_cloud(source_file.name, aligned_source)
        o3d.io.write_point_cloud(target_file.name, self.target_sampled)
        # o3d.io.write_point_cloud(target_file.name, self.target)

        source_file.close()
        target_file.close()

        subprocess.Popen([sys.executable, "view_clouds_files.py", source_file.name, target_file.name])
            

    def register_point_clouds(self):
        if self.source is None or self.target is None:
            messagebox.showerror("Load Error", "Load both source and target point clouds first.")
            return
        
        # Get values from input
        try: 
            voxel_size = float(self.voxel_entry.get())
            radius_rate = float(self.radius_entry.get())
            max_nn = int(self.max_nn_entry.get())
            max_cor_distance_rate = float(self.max_cor_distance_entry.get())
            max_iteration = int(self.max_iter_entry.get())
            confidence = float(self.confidence_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Enter valid numbers first.")

        # Preprocess
        source_down = self.source_sampled.voxel_down_sample(voxel_size)
        target_down = self.target_sampled.voxel_down_sample(voxel_size)

        source_down.estimate_normals()
        target_down.estimate_normals()

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_rate * voxel_size, max_nn=max_nn))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_rate * voxel_size, max_nn=max_nn))

        # Global registration
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=source_down,
            target=target_down,
            source_feature=source_fpfh,
            target_feature=target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=voxel_size * max_cor_distance_rate,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * max_cor_distance_rate)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, confidence)
        )
        # ICP Refinement
        # Estimate normals on down sampled clouds (required for PointToPlane ICP)
        self.source_sampled.estimate_normals()
        self.target_sampled.estimate_normals()

        # ICP Refinement
        result_icp = o3d.pipelines.registration.registration_icp(
        self.source_sampled, self.target_sampled, 0.1, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

        # Output the transformation matrix
        # self.output_text.delete('1.0', tk.END)
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "=============== Point Cloud Registration ===============")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "Transformation Matrix (ICP refined):\n")
        self.output_text.insert(tk.END, str(result_icp.transformation))
        self.transformation = result_icp.transformation

        np.set_printoptions(suppress=True)  # Disable scientific notation for Euler angles and translation

        # Convert transformation matrix into Euler angles
        euler_angles, translation = PointCloudApp.transformation_to_euler(self.transformation)

        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "Euler Angles (in degree):\n")
        self.output_text.insert(tk.END, str(euler_angles))
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "Translation:\n")
        self.output_text.insert(tk.END, str(translation))

        # Convert transformation matrix into TaitBryanAnglesZYX
        self.TaitBryanAnglesZYX = PointCloudApp.transformation_to_tait_bryan_zyx(self.transformation)

        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "TaitBryanAnglesZYX[yaw, pitch, roll] (in degree):\n")
        self.output_text.insert(tk.END, str(self.TaitBryanAnglesZYX))

        # Get alignment quantitative metric
        self.fitness = result_icp.fitness
        self.rmse = result_icp.inlier_rmse

        # Output fitness and rmse
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "Fitness:\n")
        self.output_text.insert(tk.END, str(self.fitness))
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "RMSE (Root Mean Square Error):\n")
        self.output_text.insert(tk.END, str(self.rmse))
        self.output_text.insert(tk.END, "\n")


    def remove_outliers(self,
                    keep_ratio=0.9, 
                    nb_neighbors=50,
                    std_ratio=3.5):
    
        def process(pcd):
            points = np.asarray(pcd.points)
            num_points = len(points)

            # --- Step 1: Random sampling (FAST) ---
            num_keep = int(num_points * keep_ratio)
            indices = np.random.choice(num_points, num_keep, replace=False)
            pcd_sampled = pcd.select_by_index(indices)

            # --- Step 2: Light statistical filtering ---
            _, ind = pcd_sampled.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            return pcd_sampled.select_by_index(ind)

        self.source_pcd_cleaned = process(self.source)
        self.target_pcd_cleaned = process(self.target)

        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "=============== Outlier Removal ===============")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "> Number of source point cloud before outlier removal: " + str(len(self.source.points)) + "\n")
        self.output_text.insert(tk.END, "> Number of source point cloud after outlier removal: " + str(len(self.source_pcd_cleaned.points)))
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "> Number of target point cloud before outlier removal: " + str(len(self.target.points)) + "\n")
        self.output_text.insert(tk.END, "> Number of target point cloud after outlier removal: " + str(len(self.target_pcd_cleaned.points)))
        self.output_text.insert(tk.END, "\n")

        return self.source_pcd_cleaned, self.target_pcd_cleaned
        

    @staticmethod
    def downsample_point_cloud(pcd: o3d.geometry.PointCloud, downsample_rate: float) -> o3d.geometry.PointCloud:
        """
        Downsample an Open3D point cloud by randomly keeping a specified percentage of points.

        Parameters:
            pcd (o3d.geometry.PointCloud): Input point cloud.
            keep_percentage (float): Fraction of points to keep (between 0 and 1).

        Returns:
            o3d.geometry.PointCloud: Downsampled point cloud.
        """
        if not (0 < downsample_rate <= 1):
            raise ValueError("keep_percentage must be between 0 and 1.")

        points = np.asarray(pcd.points)
        N = points.shape[0]
        num_to_keep = int(N * downsample_rate)

        # Randomly sample indices
        indices = np.random.choice(N, num_to_keep, replace=False)

        # Create a new point cloud with sampled points
        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(points[indices])

        # Optionally, copy colors if present
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            downsampled_pcd.colors = o3d.utility.Vector3dVector(colors[indices])

        # Optionally, copy normals if present
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            downsampled_pcd.normals = o3d.utility.Vector3dVector(normals[indices])

        return downsampled_pcd
    

    def downsample(self):
        try: 
            self.down_sample_rate = float(self.sample_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Enter valid numbers first.")

        self.source_sampled = PointCloudApp.downsample_point_cloud(self.source_pcd_cleaned, self.down_sample_rate)
        self.target_sampled = PointCloudApp.downsample_point_cloud(self.target_pcd_cleaned, self.down_sample_rate)
        
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "=============== Down-Sampling ===============")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "> Number of source point cloud before downsampling: " + str(np.asarray(self.source_pcd_cleaned.points).shape[0]) + "\n")
        self.output_text.insert(tk.END, "> Number of source point cloud after downsampling: " + str(np.asarray(self.source_sampled.points).shape[0]))
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "\n")
        self.output_text.insert(tk.END, "> Number of target point cloud before downsampling: " + str(np.asarray(self.target_pcd_cleaned.points).shape[0]) + "\n")
        self.output_text.insert(tk.END, "> Number of target point cloud after downsampling: " + str(np.asarray(self.target_sampled.points).shape[0]))
        self.output_text.insert(tk.END, "\n")

    @staticmethod
    def transformation_to_euler(transformation_matrix):
        """
        Converts a 4x4 transformation matrix into Euler angles and translation.
        """
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Input must be a 4x4 transformation matrix.")

        rotation_matrix = np.asarray(transformation_matrix)[:3, :3].copy()
        translation = np.asarray(transformation_matrix[:3, 3]).copy()

        # Euler angles (e.g., 'ZXZ')
        euler_angles = R.from_matrix(rotation_matrix).as_euler('zxz', degrees=True)

        return euler_angles, translation
    

    @staticmethod
    def transformation_to_tait_bryan_zyx(transformation_matrix):
        """
        Converts a 4x4 transformation matrix into Tait Bryan ZYX angles.
        """
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Input must be a 4x4 transformation matrix.")

        rotation_matrix = np.asarray(transformation_matrix)[:3, :3].copy()
        # Tait-Bryan angles (e.g., 'ZYX' => yaw-pitch-roll)
        tait_bryan = R.from_matrix(rotation_matrix).as_euler('zyx', degrees=True)

        return tait_bryan


if __name__ == "__main__":

    root = tk.Tk()
    root.title("open3d-aligner-gui")

    # Disable automatic DPI scaling
    root.tk.call("tk", "scaling", 1.0)

    # Set initial size
    root.geometry("600x800")  

    # Disable resizing
    root.resizable(False, False)

    # Lock min/max size
    root.update_idletasks()  # Let Tk calculate widget sizes
    root.minsize(600, 800)
    root.maxsize(600, 800)

    # --- Initialize the app ---
    app = PointCloudApp(root)

    root.mainloop()

