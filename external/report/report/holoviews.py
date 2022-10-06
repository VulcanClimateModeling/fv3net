import io
from pyquery import PyQuery as pq
import xml.etree.ElementTree as ET

import holoviews as hv


class HVPlot:
    """Renders holoviews plots to HTML for use in the diagnostic reports
    """

    def __init__(self, hvplot):
        self._plot = hvplot

    def __repr__(self) -> str:
        # It took hours to find this combinitation of commands!
        # it was really hard finding a combintation that
        # 1. embedded the data for an entire HoloMap object
        # 2. exported the html as a div which can easily be embedded in the reports.
        if len(self._plot) == 0:
            return ""
        r = hv.renderer("bokeh")
        html, _ = r.components(self._plot)
        html = html["text/html"]
        return html


def get_html_header() -> str:
    """Return the javascript includes needed to render holoviews plots
    """
    hv.extension("bokeh")
    hmap = hv.HoloMap()
    # need at least two plots in the holoviews for it to work
    hmap["a"] = hv.Curve([(0, 1), (0, 1)])
    hmap["b"] = hv.Curve([(0, 1), (0, 1)])

    fp = io.BytesIO()
    hv.save(hmap, fp, fmt="html")
    html = fp.getvalue().decode("UTF-8")

    # need to add root tag to parse with lxml
    doc = pq(html)
    header = ""
    for script in doc("script"):
        try:
            script.attrib["src"]
        except KeyError:
            pass
        else:
            # need to prevent self-closing script tags <src ... /> does not work in
            # Firefox (and maybe other browsers)
            header += (
                ET.tostring(script, short_empty_elements=False).decode("UTF-8").strip()
            )

    return header
