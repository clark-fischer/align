import time
import os
import numpy as np
from valis import registration
# from valis.micro_rigid_registrar import MicroRigidRegistrar # For high resolution rigid registration

slide_src_dir = "./example_datasets/ihc"
results_dst_dir = "./expected_results/registration_hi_rez"
micro_reg_fraction = 0.25 # Fraction full resolution used for non-rigid registration

# Perform high resolution rigid registration using the MicroRigidRegistrar
start = time.time()
# registrar = registration.Valis(slide_src_dir, results_dst_dir, micro_rigid_registrar_cls=MicroRigidRegistrar)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()

# Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that is 25% full resolution.
# img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
# min_max_size = np.min([np.max(d) for d in img_dims])
# img_areas = [np.multiply(*d) for d in img_dims]
# max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
# micro_reg_size = np.floor(min_max_size*micro_reg_fraction).astype(int)

# # Perform high resolution non-rigid registration using 25% full resolution
# micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=micro_reg_size)