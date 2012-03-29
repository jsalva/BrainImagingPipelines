import nipype.interfaces.freesurfer as fs
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio  
import os
import nipype.interfaces.utility as util 
from nipype.interfaces.freesurfer.preprocess import MRIConvert
from nipype.utils.filemanip import (filename_to_list, copyfiles,split_filename, list_to_filename)

#converts fsl-style Affine registration into ANTS compatible itk format
def convert_affine(brain,moving_image,bbreg_mat):
	cmd ="c3d_affine_tool -ref %s -src %s %s -fsl2ras \
		-oitk fsl2antsAffine.txt"%(brain,moving_image,bbreg_mat)
	print cmd
	os.system(cmd)
	return os.path.abspath('fsl2antsAffine.txt')

#return dimensions of input image
def image_dimensions(images):
	import nibabel as nb
	if isinstance(images,list):
		dims = []
		for image in images:
			dims.append(len(nb.load(image).get_shape()))
	else:
		dims =  len(nb.load(images).get_shape())
	
	print dims
	return dims

def first_element_of_list(_list):
	return _list[0]

#returns the workflow
def return_warpflow(base_dir):

	#inputs to workflow
	inputspec = pe.Node(
		util.IdentityInterface(
			fields=['template_file','out_fsl_files','subject_id','brain',
				'segmentation','func_files','mean_func']),
		name='inputspec')

	#converts brain from freesurfer mgz into nii
	brain_2nii = pe.Node(
		fs.preprocess.MRIConvert(),
		name='brain_2nii')
	brain_2nii.inputs.out_type='nii'

	#converts freesurfer segmentation into nii 
	aparcaseg_2nii = pe.Node(
		fs.preprocess.MRIConvert(),
		name='aparcaseg_2nii')
	aparcaseg_2nii.inputs.out_type='nii'

	#create mask excluding everything outside of cortical surface (from 
	#freesurfer segmentation file)
	create_mask = pe.Node(
		fs.model.Binarize(),
		name='create_mask')
	create_mask.inputs.min=1
	create_mask.inputs.out_type = 'nii'

	#apply mask to anatomical
	apply_mask = pe.Node(
		fs.utils.ApplyMask(),
		name='apply_mask')

	#makes fsl-style coregistration ANTS compatible
	fsl_registration_to_itk = pe.Node(
		util.Function(
			input_names=['brain','moving_image','bbreg_mat'],
			output_names=['fsl2antsAffine'],
			function=convert_affine),
		name='fsl_registration_to_itk')
	
	#use ANTS to warp the masked anatomical image to a template image
	warp_brain = pe.Node(
		ants.GenWarpFields(), 
		name='warp_brain')

	#collects series of transformations to be applied to the moving images
	collect_transforms = pe.Node(
		util.Merge(3),
		name='collect_transforms')	

	#performs series of transformations on moving images
	warp_images = pe.MapNode(
		ants.WarpTimeSeriesImageMultiTransform(),
		name='warp_images',
		iterfield=['moving_image','dimension'])

	#collects workflow outputs
	outputspec = pe.Node(
		util.IdentityInterface(
			fields=['warped_images','affine_transformation','warp_field',
				'inverse_warp','warped_brain']),
		name='outputspec')

	#initializes and connects workflow nodes
	normalize = pe.Workflow(name='normalize')
	normalize.base_dir = base_dir
	normalize.connect([
		(inputspec,warp_brain,[('template_file','reference_image')]),
		(inputspec,warp_images,[('template_file','reference_image')]),
		(inputspec,brain_2nii,[('brain','in_file')]),
		(inputspec,aparcaseg_2nii,[('segmentation','in_file')]),
		(aparcaseg_2nii,create_mask,[('out_file','in_file')]),
		(create_mask,apply_mask,[('binary_file','mask_file')]),
		(brain_2nii,apply_mask,[('out_file','in_file')]),
		(apply_mask,warp_brain,[('out_file','input_image')]),
		(apply_mask,fsl_registration_to_itk,[('out_file','brain')]),
		(inputspec,fsl_registration_to_itk,[('out_fsl_files','bbreg_mat')]),
		(inputspec,fsl_registration_to_itk,[('mean_func','moving_image')]),
		(fsl_registration_to_itk,collect_transforms,[('fsl2antsAffine','in3')]),
		(warp_brain,collect_transforms,[('warp_field','in1'),
			('affine_transformation','in2')]),
		(inputspec,warp_images,[('func_files','moving_image')]),
		(inputspec,warp_images,[(('func_files',image_dimensions),'dimension')]),
		(collect_transforms,warp_images,[(('out',first_element_of_list),'transformation_series')]),
		(warp_images,outputspec,[('output_images','warped_images')]),
		(warp_brain,outputspec,[('affine_transformation','affine_transformation'),
			('warp_field','warp_field'),
			('inverse_warp_field','inverse_warp'),
			('output_file','warped_brain')])])

	return normalize
