from fast_dash import fastdash
import matplotlib.pyplot as plt
import yt 


@fastdash
def slice_plot(sample_dataset_name: str,
               axis: str,
               field_type: str = 'gas',
               field_name: str = 'density') -> plt.Figure:

    # cleanup from inputs
    sample_dataset_name = sample_dataset_name.strip()
    axis = axis.strip()
    field_type = field_type.strip()
    field_name = field_name.strip()

    # do a thing, return a matlab figure handle
    ds = yt.load_sample(sample_dataset_name)
    fld = (field_type, field_name)
    slc = yt.SlicePlot(ds, axis, fld)
    slc.render()
    return slc.plots[fld].figure