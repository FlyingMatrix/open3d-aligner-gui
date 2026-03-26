import sys
import open3d as o3d
import os

if __name__ == "__main__":
    source_file = sys.argv[1]
    target_file = sys.argv[2]

    # Load point clouds
    source = o3d.io.read_point_cloud(source_file)
    target = o3d.io.read_point_cloud(target_file)

    # Set colors - source: Amber, target: Cyan
    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])

    # Display
    o3d.visualization.draw_geometries([source, target],
                                      window_name="Point Cloud Viewer (Source: Amber, Target: Cyan)",
                                      width=1200, height=900)

    # Clean up temporary files after the window closes
    os.remove(source_file)
    os.remove(target_file)