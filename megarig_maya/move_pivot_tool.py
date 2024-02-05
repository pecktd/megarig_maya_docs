# Maya modules
import maya.cmds as mc
import maya.OpenMaya as om


def dag_path_from_str(dag_str: str) -> om.MDagPath:
    sellist = om.MSelectionList()
    sellist.add(dag_str)
    mdag_path = om.MDagPath()
    sellist.getDagPath(0, mdag_path)

    return mdag_path


def get_offset_translation(
    matrix: om.MTransformationMatrix, inverse_matrix: om.MTransformationMatrix
) -> om.MVector:
    result_mat = matrix * inverse_matrix
    return om.MVector(
        om.MScriptUtil.getDoubleArrayItem(result_mat[3], 0),
        om.MScriptUtil.getDoubleArrayItem(result_mat[3], 1),
        om.MScriptUtil.getDoubleArrayItem(result_mat[3], 2),
    )


def move_pivot(dag_name: str, pivot_name: str) -> None:
    """Match the rotate pivot of the given object to the position of the given
    pivot node.
    """
    ct = mc.currentTime(q=True)

    dag = dag_path_from_str(dag_name)
    piv = dag_path_from_str(pivot_name)

    mc.setKeyframe(dag.fullPathName(), t=[ct - 1, ct], i=True, an=True)

    piv_vec = get_offset_translation(
        piv.inclusiveMatrix(), dag.inclusiveMatrixInverse()
    )

    pre_mod_vec = get_offset_translation(
        dag.inclusiveMatrix(), dag.exclusiveMatrixInverse()
    )

    mc.setKeyframe(dag.fullPathName(), at="rp", t=[ct - 1], ott="step")
    mc.setAttr(dag.fullPathName() + ".rp", *piv_vec)
    mc.setKeyframe(dag.fullPathName(), at="rp", ott="step")

    mod_vec = get_offset_translation(
        dag.inclusiveMatrix(), dag.exclusiveMatrixInverse()
    )
    offset_rpt_vec = pre_mod_vec - mod_vec

    rpt = om.MVector(*mc.getAttr(dag.fullPathName() + ".rpt")[0])

    mc.setKeyframe(dag.fullPathName(), at="rpt", t=[ct - 1], ott="step")
    mc.setAttr(dag.fullPathName() + ".rpt", *(rpt + offset_rpt_vec))
    mc.setKeyframe(dag.fullPathName(), at="rpt", ott="step")


def move_selected_pivot() -> None:
    """Select pivot node then a transform node.
    Script matches the rotate pivot of the selected transform node to
    the selected pivot node.
    """
    sels = mc.ls(sl=True)
    move_pivot(sels[1], sels[0])
