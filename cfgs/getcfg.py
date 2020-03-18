'''
Function:
	used to get config file for specified dataset and backbone.
Author:
	Charles
'''
def getCfgByDatasetAndBackbone(datasetname, backbonename):
	if [datasetname, backbonename] == ['coco', 'darknet53']:
		import cfgs.cfg_coco_darknet53 as cfg
		cfg_file_path = 'cfgs/cfg_coco_darknet53'
	else:
		raise ValueError('Can not find cfg file for dataset <%s> and backbone <%s>...' % (datasetname, backbonename))
	return cfg, cfg_file_path