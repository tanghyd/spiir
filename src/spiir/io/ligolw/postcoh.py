import logging
from typing import Union
from xml.sax.xmlreader import AttributesImpl

import glue.ligolw.lsctables
from lal import LIGOTimeGPS
from ligo.lw import array, ligolw, lsctables, param, table

logger = logging.getLogger(__name__)

# TO DO:
# - Port compatibility with C postcoh table
# - Check compatibility with dbtables

# database compatibility for ilwd:char - review this
# from glue.ligolw import dbtables
# dbtables.ligolwtypes.ToPyType["ilwd:char"] = six.text_type

# Mapping to update PostcohInspiralTable columns from legacy to latest named versions.
_IFOS = ("H1", "L1", "V1", "K1")
_LEGACY_COLUMN_MAP = {
    **{f"chisq_{ifo[0]}": f"chisq_{ifo}" for ifo in _IFOS},
    **{f"coaphase_{ifo[0]}": f"coaphase_{ifo}" for ifo in _IFOS},
    **{f"deff_{ifo[0]}": f"deff_{ifo}" for ifo in _IFOS},
    **{f"end_time_{ifo[0]}": f"end_time_sngl_{ifo}" for ifo in _IFOS},
    **{f"end_time_ns_{ifo[0]}": f"end_time_ns_sngl_{ifo}" for ifo in _IFOS},
    **{f"far_{ifo[0].lower()}": f"far_sngl_{ifo}" for ifo in _IFOS},
    **{f"far_{ifo[0].lower()}_2h": f"far_2h_sngl_{ifo}" for ifo in _IFOS},
    **{f"far_{ifo[0].lower()}_1d": f"far_1d_sngl_{ifo}" for ifo in _IFOS},
    **{f"far_{ifo[0].lower()}_1w": f"far_1w_sngl_{ifo}" for ifo in _IFOS},
    **{f"snglsnr_{ifo[0]}": f"snglsnr_{ifo}" for ifo in _IFOS},
}
_LEGACY_COLUMN_MAP.update(
    {f"postcoh:{old}": f"postcoh:{new}" for old, new in _LEGACY_COLUMN_MAP.items()}
)


def rename_legacy_postcoh_columns(xmldoc: ligolw.Document) -> ligolw.Document:
    """Updates legacy PostcohInspiralTable table column names to their latest versions.

    Parameters
    ----------
    xmldoc: ligo.lw.ligo.lw.ligolw.Document
        A valid LIGO_LW XML Document element with a PostcohInspiralTable element.

    Returns
    -------
    ligo.lw.ligolw.Document
        The same LIGO_LW Document object with updated table column names.
    """
    for elem in xmldoc.getElements(lambda e: e.tagName == table.Table.tagName):
        if elem.tagName == table.Table.tagName:
            if elem.Name == "postcoh":
                valid_columns = lsctables.TableByName[elem.Name].validcolumns
                for column in elem.getElementsByTagName(ligolw.Column.tagName):
                    name = column.getAttribute("Name")
                    if name not in valid_columns and name in _LEGACY_COLUMN_MAP:
                        column.setAttribute("Name", _LEGACY_COLUMN_MAP[name])

                # update named column attributes for every row instance
                for row in elem:
                    for k in row.__dict__.copy():
                        if k in _LEGACY_COLUMN_MAP:
                            row.__dict__[_LEGACY_COLUMN_MAP[k]] = row.__dict__.pop(k)
    return xmldoc


def include_missing_postcoh_columns(
    xmldoc: ligolw.Document,
    nullable: bool = True,
) -> ligolw.Document:
    """Adds any missing PostcohInspiralTable columns and fills them with default values.

    NOTE: This function assumes that all legacy PostcohInspiralTable columns have been
    updated to their latest values - otherwise unconverted column names will be
    interpreted as if they were simply missing.

    Parameters
    ----------
    xmldoc: ligo.lw.ligo.lw.ligolw.Document
        A valid LIGO_LW XML Document element with a PostcohInspiralTable element.
    nullable: bool
        If True, sets the values for missing postcoh columns to NoneType,
        otherwise it is set to an appropriate default value given the column type.

    Returns
    -------
    ligo.lw.ligolw.Document
        The same LIGO_LW Document object with updated table column names.
    """

    def default_value(name: str, dtype: str) -> Union[float, int, str]:
        """Helper function to fill a sensible default value to a given column type."""
        if dtype in ["real_4", "real_8"]:
            return 0.0
        elif dtype in ["int_4s", "int_8s"]:  # this case includes row IDs
            return 0
        elif dtype in ["char_s", "char_v", "string", "lstring"]:
            return ""
        else:
            raise NotImplementedError(f"Cannot initialize col {name} of type {dtype}.")

    for elem in xmldoc.getElements(lambda e: e.tagName == table.Table.tagName):
        if elem.tagName == table.Table.tagName:
            if elem.Name == "postcoh":
                # insert missing column to postcoh table columns in order of valid cols
                valid_columns = lsctables.TableByName[elem.Name].validcolumns
                present_columns = elem.getElementsByTagName(table.Column.tagName)
                present_column_names = [col.Name for col in present_columns]
                missing_columns = []
                for name in valid_columns:
                    if name not in present_column_names:
                        column_type = valid_columns[name]
                        attrs = {"Name": "%s" % name, "Type": column_type}
                        column = table.Column(AttributesImpl(attrs))
                        streams = elem.getElementsByTagName(ligolw.Stream.tagName)
                        if streams:
                            elem.insertBefore(column, streams[0])
                        else:
                            elem.appendChild(column)
                        missing_columns.append(name)

                # update values for new columns for every row instance
                for row in elem:
                    for name in missing_columns:
                        if nullable:
                            value = None
                        else:
                            value = default_value(name, valid_columns[name])
                        setattr(row, name, value)

    return xmldoc


PostcohInspiralID = table.next_id.type("event_id")


class PostcohInspiralTable(table.Table):
    tableName = "postcoh"
    validcolumns = {
        "process_id": "int_8s",
        "event_id": "int_8s",
        "end_time": "int_4s",
        "end_time_ns": "int_4s",
        "end_time_sngl_H1": "int_4s",
        "end_time_ns_sngl_H1": "int_4s",
        "end_time_sngl_L1": "int_4s",
        "end_time_ns_sngl_L1": "int_4s",
        "end_time_sngl_V1": "int_4s",
        "end_time_ns_sngl_V1": "int_4s",
        **{
            key: value
            for end_times in (
                {f"end_time_sngl_{ifo}": "int_4s", f"end_time_ns_sngl_{ifo}": "int_4s"}
                for ifo in _IFOS
            )
            for key, value in end_times.items()
        },
        **{f"snglsnr_{ifo}": "real_4" for ifo in _IFOS},
        **{f"coaphase_{ifo}": "real_4" for ifo in _IFOS},
        **{f"chisq_{ifo}": "real_4" for ifo in _IFOS},
        "is_background": "int_4s",
        "livetime": "int_4s",
        "livetime_1w": "int_4s",
        "livetime_1d": "int_4s",
        "livetime_2h": "int_4s",
        "nevent_1w": "int_4s",
        "nevent_1d": "int_4s",
        "nevent_2h": "int_4s",
        **{f"livetime_1w_sngl_{ifo}": "int_4s" for ifo in _IFOS},
        **{f"livetime_1d_sngl_{ifo}": "int_4s" for ifo in _IFOS},
        **{f"livetime_2h_sngl_{ifo}": "int_4s" for ifo in _IFOS},
        **{f"nevent_1w_sngl_{ifo}": "int_4s" for ifo in _IFOS},
        **{f"nevent_1d_sngl_{ifo}": "int_4s" for ifo in _IFOS},
        **{f"nevent_2h_sngl_{ifo}": "int_4s" for ifo in _IFOS},
        "ifos": "lstring",
        "pivotal_ifo": "lstring",
        "tmplt_idx": "int_4s",
        "bankid": "int_4s",
        "pix_idx": "int_4s",
        "cohsnr": "real_4",
        "nullsnr": "real_4",
        "cmbchisq": "real_4",
        "spearman_pval": "real_4",
        "fap": "real_4",
        "far": "real_4",
        "far_2h": "real_4",
        "far_1d": "real_4",
        "far_1w": "real_4",
        **{f"far_sngl_{ifo}": "real_4" for ifo in _IFOS},
        **{f"far_1w_sngl_{ifo}": "real_4" for ifo in _IFOS},
        **{f"far_1d_sngl_{ifo}": "real_4" for ifo in _IFOS},
        **{f"far_2h_sngl_{ifo}": "real_4" for ifo in _IFOS},
        "skymap_fname": "lstring",
        "template_duration": "real_8",
        "mass1": "real_4",
        "mass2": "real_4",
        "mchirp": "real_4",
        "mtotal": "real_4",
        "spin1x": "real_4",
        "spin1y": "real_4",
        "spin1z": "real_4",
        "spin2x": "real_4",
        "spin2y": "real_4",
        "spin2z": "real_4",
        "eta": "real_4",
        "f_final": "real_4",
        "ra": "real_8",
        "dec": "real_8",
        **{f"deff_{ifo}": "real_8" for ifo in _IFOS},
        "rank": "real_8",
    }
    constraints = "PRIMARY KEY (event_id)"
    next_id = PostcohInspiralID(0)


class PostcohInspiral(table.Table.RowType):
    __slots__ = tuple(map(table.Column.ColumnName, PostcohInspiralTable.validcolumns))

    @property
    def end(self):
        if self.end_time is None and self.end_time_ns is None:
            return None
        return LIGOTimeGPS(self.end_time, self.end_time_ns)

    @end.setter
    def end(self, gps):
        if gps is None:
            self.end_time = self.end_time_ns = None
        else:
            self.end_time, self.end_time_ns = gps.gpsSeconds, gps.gpsNanoSeconds


PostcohInspiralTable.RowType = PostcohInspiral

# add our custom postcoh table to the lsctables.TableByName dict
TableByName = {PostcohInspiralTable.tableName: PostcohInspiralTable}
lsctables.TableByName.update(TableByName)
glue.ligolw.lsctables.TableByName.update(TableByName)  # legacy compatibility


def use_in(ContentHandler):
    """Modify ContentHandler, a sub-class of ligo.lw.ligolw.LIGOLWContentHandler, to
    cause it to use the Table classes defined in this module when parsing XML documents.

    NOTE: This function has not been tested and should not be assumed to work correctly.

    Examples
    --------
    >>> from glue.ligolw import ligolw
    >>> class MyContentHandler(ligolw.LIGOLWContentHandler):
    ...	pass
    ...
    >>> use_in(MyContentHandler)
    <class 'glue.ligolw.lsctables.MyContentHandler'>

    """
    ContentHandler = array.use_in(ContentHandler)
    ContentHandler = param.use_in(ContentHandler)
    ContentHandler = table.use_in(ContentHandler)
    ContentHandler = lsctables.use_in(ContentHandler)

    def startTable(self, parent, attrs, __orig_startTable=ContentHandler.startTable):
        name = table.Table.TableName(attrs["Name"])
        if name in TableByName:
            return TableByName[name](attrs)
        return __orig_startTable(self, parent, attrs)

    ContentHandler.startTable = startTable
    return ContentHandler
