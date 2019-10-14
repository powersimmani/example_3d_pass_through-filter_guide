# Pass thourgh filter and its guidline for python

This is a code example for applying pass through filter which open3d library does not have. A pass through filter is a filter that extracts only points within a specified range and can be used to preprocess the raw point cloud. The amount of computation in a typical three-dimensional algorithm is proportional to the square of the minimum number of points. This filter is beneficial for reducing the amount of computation.

When specifying the range of pass through filters, it can be difficult to know which part of the raw point cloud should be extracted. I have implemented a guideline function to help with this.

# Requirements
- Numpy 
- [Open3d](https://github.com/intel-isl/Open3D)

# Examples

## 1.Raw point cloud
<img src="https://github.com/powersimmani/example_3d_pass_through-filter_guide/blob/master/images/1.raw_point_cloud.PNG?raw=true" width="384" height="216">


## 2.Draw filter guidelines
<img src="https://github.com/powersimmani/example_3d_pass_through-filter_guide/blob/master/images/2.draw_filter_guidelines.PNG?raw=true" width="384" height="216">

## 3.Point cloud filtered with guildline
<img src="https://github.com/powersimmani/example_3d_pass_through-filter_guide/blob/master/images/3.filtered_guildline.PNG?raw=true" width="384" height="216">


## 4.Point cloud filtered without guildline
<img src="https://github.com/powersimmani/example_3d_pass_through-filter_guide/blob/master/images/4.only_filtered.PNG?raw=true" width="384" height="216">


# Useful References
This code can be usefully connected or extended with the following example code.

- [Point cloud streaming using PyKinect2](https://github.com/powersimmani/example_3d_reconstruction_pykinect2)
