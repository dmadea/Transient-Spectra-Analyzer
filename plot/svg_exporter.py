from .Exporter import Exporter
from ..python2_3 import asUnicode
from ..parametertree import Parameter
from ..Qt import QtGui, QtCore, QtSvg, USE_PYSIDE
from .. import debug
from .. import functions as fn
import re
import xml.dom.minidom as xml
import numpy as np

import pyqtgraph.exporters.SVGExporter



def _generateItemSvg(item, nodes=None, root=None):
    ## This function is intended to work around some issues with Qt's SVG generator
    ## and SVG in general.
    ## 1) Qt SG does not implement clipping paths. This is absurd.
    ##    The solution is to let Qt generate SVG for each item independently,
    ##    then glue them together manually with clipping.
    ##
    ##    The format Qt generates for all items looks like this:
    ##
    ##    <g>
    ##        <g transform="matrix(...)">
    ##            one or more of: <path/> or <polyline/> or <text/>
    ##        </g>
    ##        <g transform="matrix(...)">
    ##            one or more of: <path/> or <polyline/> or <text/>
    ##        </g>
    ##        . . .
    ##    </g>
    ##
    ## 2) There seems to be wide disagreement over whether path strokes
    ##    should be scaled anisotropically.
    ##      see: http://web.mit.edu/jonas/www/anisotropy/
    ##    Given that both inkscape and illustrator seem to prefer isotropic
    ##    scaling, we will optimize for those cases.
    ##
    ## 3) Qt generates paths using non-scaling-stroke from SVG 1.2, but
    ##    inkscape only supports 1.1.
    ##
    ##    Both 2 and 3 can be addressed by drawing all items in world coordinates.

    profiler = debug.Profiler()

    if nodes is None:  ## nodes maps all node IDs to their XML element.
        ## this allows us to ensure all elements receive unique names.
        nodes = {}

    if root is None:
        root = item

    ## Skip hidden items
    if hasattr(item, 'isVisible') and not item.isVisible():
        return None

    ## If this item defines its own SVG generator, use that.
    if hasattr(item, 'generateSvg'):
        return item.generateSvg(nodes)

    ## Generate SVG text for just this item (exclude its children; we'll handle them later)
    tr = QtGui.QTransform()
    if isinstance(item, QtGui.QGraphicsScene):
        xmlStr = "<g>\n</g>\n"
        doc = xml.parseString(xmlStr)
        childs = [i for i in item.items() if i.parentItem() is None]
    elif item.__class__.paint == QtGui.QGraphicsItem.paint:
        xmlStr = "<g>\n</g>\n"
        doc = xml.parseString(xmlStr)
        childs = item.childItems()
    else:
        childs = item.childItems()
        tr = itemTransform(item, item.scene())

        ## offset to corner of root item
        if isinstance(root, QtGui.QGraphicsScene):
            rootPos = QtCore.QPoint(0, 0)
        else:
            rootPos = root.scenePos()
        tr2 = QtGui.QTransform()
        tr2.translate(-rootPos.x(), -rootPos.y())
        tr = tr * tr2

        arr = QtCore.QByteArray()
        buf = QtCore.QBuffer(arr)
        svg = QtSvg.QSvgGenerator()
        svg.setOutputDevice(buf)
        # dpi = QtGui.QDesktopWidget().physicalDpiX()
        dpi = QtGui.QDesktopWidget().logicalDpiX()
        svg.setResolution(dpi)

        p = QtGui.QPainter()
        p.begin(svg)
        if hasattr(item, 'setExportMode'):
            item.setExportMode(True, {'painter': p})
        try:
            p.setTransform(tr)
            item.paint(p, QtGui.QStyleOptionGraphicsItem(), None)
        finally:
            p.end()
            ## Can't do this here--we need to wait until all children have painted as well.
            ## this is taken care of in generateSvg instead.
            # if hasattr(item, 'setExportMode'):
            # item.setExportMode(False)

        if USE_PYSIDE:
            xmlStr = str(arr)
        else:
            xmlStr = bytes(arr).decode('utf-8')
        doc = xml.parseString(xmlStr)

    try:
        ## Get top-level group for this item
        g1 = doc.getElementsByTagName('g')[0]
        ## get list of sub-groups
        g2 = [n for n in g1.childNodes if isinstance(n, xml.Element) and n.tagName == 'g']

        defs = doc.getElementsByTagName('defs')
        if len(defs) > 0:
            defs = [n for n in defs[0].childNodes if isinstance(n, xml.Element)]
    except:
        print(doc.toxml())
        raise

    profiler('render')

    ## Get rid of group transformation matrices by applying
    ## transformation to inner coordinates
    correctCoordinates(g1, defs, item)
    profiler('correct')
    ## make sure g1 has the transformation matrix
    # m = (tr.m11(), tr.m12(), tr.m21(), tr.m22(), tr.m31(), tr.m32())
    # g1.setAttribute('transform', "matrix(%f,%f,%f,%f,%f,%f)" % m)

    # print "=================",item,"====================="
    # print g1.toprettyxml(indent="  ", newl='')

    ## Inkscape does not support non-scaling-stroke (this is SVG 1.2, inkscape supports 1.1)
    ## So we need to correct anything attempting to use this.
    # correctStroke(g1, item, root)

    ## decide on a name for this item
    baseName = item.__class__.__name__
    i = 1
    while True:
        name = baseName + "_%d" % i
        if name not in nodes:
            break
        i += 1
    nodes[name] = g1
    g1.setAttribute('id', name)

    ## If this item clips its children, we need to take care of that.
    childGroup = g1  ## add children directly to this node unless we are clipping
    if not isinstance(item, QtGui.QGraphicsScene):
        ## See if this item clips its children
        if int(item.flags() & item.ItemClipsChildrenToShape) > 0:
            ## Generate svg for just the path
            # if isinstance(root, QtGui.QGraphicsScene):
            # path = QtGui.QGraphicsPathItem(item.mapToScene(item.shape()))
            # else:
            # path = QtGui.QGraphicsPathItem(root.mapToParent(item.mapToItem(root, item.shape())))
            path = QtGui.QGraphicsPathItem(item.mapToScene(item.shape()))
            item.scene().addItem(path)
            try:
                # pathNode = _generateItemSvg(path, root=root).getElementsByTagName('path')[0]
                pathNode = _generateItemSvg(path, root=root)[0].getElementsByTagName('path')[0]
                # assume <defs> for this path is empty.. possibly problematic.
            finally:
                item.scene().removeItem(path)

            ## and for the clipPath element
            clip = name + '_clip'
            clipNode = g1.ownerDocument.createElement('clipPath')
            clipNode.setAttribute('id', clip)
            clipNode.appendChild(pathNode)
            g1.appendChild(clipNode)

            childGroup = g1.ownerDocument.createElement('g')
            childGroup.setAttribute('clip-path', 'url(#%s)' % clip)
            g1.appendChild(childGroup)
    profiler('clipping')

    ## Add all child items as sub-elements.
    childs.sort(key=lambda c: c.zValue())
    for ch in childs:
        csvg = _generateItemSvg(ch, nodes, root)
        if csvg is None:
            continue
        cg, cdefs = csvg
        childGroup.appendChild(cg)  ### this isn't quite right--some items draw below their parent (good enough for now)
        defs.extend(cdefs)

    profiler('children')
    return g1, defs
