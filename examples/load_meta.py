from cichlidanalysis.io.meta import load_yaml

# meta = load_yaml('.', 'roi')
meta = load_yaml('/Users/annikanichols/Documents/cichlid-analysis/examples', 'roi_file')

print(meta)
#
# meta = load_yaml('/Users/annikanichols/Documents/cichlid-analysis/examples/roi_file.yaml')
# meta = load_yaml('./roi_file.yaml')