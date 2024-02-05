# Python modules
from typing import Union, List, Tuple
import math
from functools import partial

# Maya modules
import maya.OpenMaya as om
import maya.cmds as mc

# megarig modules
from megarig_maya import core as mrc
from megarig_maya import naming_tools as mrnt


# Max/Min values to be used on translation.
MAX_T = 1000
MIN_T = -1000


def connect_trs(dag_a: mrc.Dag, dag_b: mrc.Dag):
    """Connects translate, rotate and scale attributes from the given
    daga to dagb.
    """
    for attr in ("t", "r", "s"):
        dag_a.attr(attr) >> dag_b.attr(attr)


def mvec_to_tuple(mvec: om.MVector) -> tuple:
    """Converts MVector to tuple."""
    return (mvec.x, mvec.y, mvec.z)


def distance(dag1: mrc.Dag, dag2: mrc.Dag) -> float:
    return (om.MVector(*dag2.world_pos) - om.MVector(*dag1.world_pos)).length()


def create_vector_node(
    in1: mrc.Attr, in2: mrc.Attr
) -> Tuple[mrc.Node, mrc.Node]:
    """Creates a normalized vector from in1 to in2"""
    sub = mrc.create("plusMinusAverage")
    sub.attr("op").v = mrc.Op.subtract

    norm = mrc.create("vectorProduct")
    norm.attr("op").v = mrc.Op.no
    norm.attr("no").v = True

    in2 >> sub.attr("i3[0]")
    in1 >> sub.attr("i3[1]")
    sub.attr("o3") >> norm.attr("i1")

    return sub, norm


def selected_edges_to_eyelid_curve(name: str) -> mrc.Dag:
    """Converts selected poly edges to an eyelid curve."""
    crv = mrc.Dag(mc.polyToCurve(ch=False, form=0, degree=1)[0])
    crvshape = crv.get_shape()
    crvspans = crvshape.attr("spans").v

    cv0pos = mc.xform("{}.cv[0]".format(crv.name), q=True, t=True, ws=True)
    cvlastpos = mc.xform(
        "{}.cv[{}]".format(crv.name, crvspans), q=True, t=True, ws=True
    )
    if abs(cv0pos[0]) > abs(cvlastpos[0]):
        mc.reverseCurve(crv.name, ch=False)

    if name:
        side = mrc.Side.left
        if cv0pos[0] < 0:
            side = mrc.Side.right

        crv.name = mrnt.compose(name, None, side, "TmpCrv")

    return crv


def remove_connected_constraints(dag: Union[mrc.Dag, str]):
    """Removes all constraint nodes connected to the given dag."""
    mc.delete(mc.listConnections(dag, s=True, d=False, type="constraint"))


def curve_guide(root: mrc.Dag, end: mrc.Dag):
    """Creates a NURBs curve guide from the root_dag to the end_dag."""

    def _rename_handle_shape(handle: mrc.Dag):
        shape = mc.listRelatives(
            handle.name, f=True, s=True, type="clusterHandle"
        )[0]
        mc.rename(shape, "{}ClusterHandleShape".format(handle.name))

    curve = mrc.Dag(mc.curve(d=1, p=[(0, 0, 0), (0, 0, 0)]))

    root_clust, root_clust_handle = mrc.cluster(
        "{}.cv[0]".format(curve.name), wn=(root.name, root.name)
    )
    _rename_handle_shape(root)

    end_clust, end_clust_handle = mrc.cluster(
        "{}.cv[1]".format(curve.name), wn=(end.name, end.name)
    )
    _rename_handle_shape(end)

    curve.attr("overrideEnabled").v = True
    curve.attr("overrideDisplayType").v = 2
    curve.attr("inheritsTransform").v = False

    return curve, root_clust, root_clust_handle, end_clust, end_clust_handle


def orient_bone(jnt: mrc.Dag, up_axis: tuple, up_vec: tuple):
    """Aims Y axis of the given joint to the first child node."""
    parnode = jnt.get_parent()
    chdjnts = jnt.get_all_children()

    def _reset_ori(jnt: mrc.Dag):
        jnt.attr("r").v = (0, 0, 0)
        jnt.attr("jo").v = (0, 0, 0)

    if chdjnts:
        for chdjnt in chdjnts:
            # Parents all child joints to world.
            chdjnt.set_parent(None)
        jnt.set_parent(None)

        aim(jnt, chdjnts[0], (0, 1, 0), up_axis, up_vec)

        for chdjnt in chdjnts:
            # Reparents all child joints.
            chdjnt.set_parent(jnt)

            if chdjnt.m_object.apiType() != om.MFn.kJoint:
                # Skips if current node is not joint.
                continue

            # If current child joint doesn't have grandchild joints,
            # resets its orientation.
            if not chdjnt.get_child():
                _reset_ori(chdjnt)
    else:
        _reset_ori(jnt)

    jnt.set_parent(parnode)
    mc.select(jnt.name, r=True)


def get_rotation_quaternion(dag: mrc.Dag) -> om.MQuaternion:
    """Gets quaternion rotation from the given dag."""
    xformfn = om.MFnTransform(dag.m_dag_path)

    utilx = om.MScriptUtil(0.0)
    ptrx = utilx.asDoublePtr()

    utily = om.MScriptUtil(0.0)
    ptry = utily.asDoublePtr()

    utilz = om.MScriptUtil(0.0)
    ptrz = utilz.asDoublePtr()

    utilw = om.MScriptUtil(0.0)
    ptrw = utilw.asDoublePtr()

    xformfn.getRotationQuaternion(ptrx, ptry, ptrz, ptrw, om.MSpace.kWorld)

    return om.MQuaternion(
        utilx.getDouble(ptrx),
        utily.getDouble(ptry),
        utilz.getDouble(ptrz),
        utilw.getDouble(ptrw),
    )


def get_offset_rotate(dag: mrc.Dag, target: mrc.Dag) -> tuple:
    """Gets offset rotation values from 'dag' to 'target'.
    Returns a tuple of euler rotation angles.
    """
    dagquat = get_rotation_quaternion(dag)
    dagquat_inv = dagquat.inverse()
    tarquat = get_rotation_quaternion(target)

    ofstquat = tarquat * dagquat_inv
    ofsteul = ofstquat.asEulerRotation()
    ofsteul.reorderIt(dag.attr("rotateOrder").v)

    return tuple(
        [math.degrees(angle) for angle in [ofsteul.x, ofsteul.y, ofsteul.z]]
    )


def add_vector_attr(node: mrc.Node, attr_prefix: str, k: bool) -> mrc.Attr:
    """Adds a vector attribute with the given attr_prefix to the given node."""
    node.add(ln=attr_prefix, at="double3", k=k)
    node.add(ln="{}X".format(attr_prefix), at="double", p=attr_prefix, k=k)
    node.add(ln="{}Y".format(attr_prefix), at="double", p=attr_prefix, k=k)
    node.add(ln="{}Z".format(attr_prefix), at="double", p=attr_prefix, k=k)

    return node.attr(attr_prefix)


def mult_by_parent_inverse_mat(
    dag: mrc.Dag, driver: mrc.Attr, t: bool, r: bool, s: bool
) -> Tuple[mrc.Node, mrc.Node]:
    """Uses multMatrix node to multiply between the driver(A matrix attribute)
    and the dag's parentInverseMatrix.
    decomposeMatrix is also created to get the transformation output.

    Returns multMatrix and decomposeMatrix nodes.
    """
    multmat = mrc.create("multMatrix")
    decomp = mrc.create("decomposeMatrix")

    driver >> multmat.attr("matrixIn").last()
    dag.attr("parentInverseMatrix[0]") >> multmat.attr("matrixIn").last()
    multmat.attr("matrixSum") >> decomp.attr("inputMatrix")
    dag.attr("rotateOrder") >> decomp.attr("inputRotateOrder")

    if t:
        decomp.attr("outputTranslate") >> dag.attr("t")
    if r:
        decomp.attr("outputRotate") >> dag.attr("r")
    if s:
        decomp.attr("outputScale") >> dag.attr("s")

    return multmat, decomp


def swing_twist_nodes(
    axis: str,
) -> Tuple[mrc.Node, mrc.Node, mrc.Node, mrc.Node, mrc.Node]:
    """Adds swing/twist node connections.

    Returns eulerToQuat, quatToEuler(twist), quatToEuler(swing), quatInvert
    and quatProd nodes.
    """
    e2q = mrc.create("eulerToQuat")
    q2e_tw = mrc.create("quatToEuler")
    q2e_sw = mrc.create("quatToEuler")
    quat_inv = mrc.create("quatInvert")
    quat_prod = mrc.create("quatProd")

    in_quat = "inputQuat{}".format(axis.upper())
    out_quat = "outputQuat{}".format(axis.upper())

    # Twist
    e2q.attr(out_quat) >> q2e_tw.attr(in_quat)
    e2q.attr("outputQuatW") >> q2e_tw.attr("inputQuatW")

    # Swing
    e2q.attr(out_quat) >> quat_inv.attr(in_quat)
    e2q.attr("outputQuatW") >> quat_inv.attr("inputQuatW")
    quat_inv.attr("outputQuat") >> quat_prod.attr("input1Quat")
    e2q.attr("outputQuat") >> quat_prod.attr("input2Quat")
    quat_prod.attr("outputQuat") >> q2e_sw.attr("inputQuat")

    return e2q, q2e_tw, q2e_sw, quat_inv, quat_prod


def add_non_roll_joint(
    dag: mrc.Dag, axis: str, parent: mrc.Dag
):
    """Adds non-roll joint to the given dag and parent the created node
    to the given parent.
    The type of created node is defined by the given dag_type.

    Returns A non-roll joint, pointConstraint, orientConstraint, eulerToQuat,
    quatToEuler(twist), quatToEuler(swing), quatInvert and quatProd nodes.
    """
    nonroll = create_joint_at(dag)
    if parent:
        nonroll.set_parent(parent)

    pointcons = mrc.pntcons(dag, nonroll)
    oricons = mrc.oricons(dag, nonroll)

    if axis == "y":
        nonroll.attr("ro").v = mrc.Ro.yzx
    elif axis == "x":
        nonroll.attr("ro").v = mrc.Ro.xyz
    elif axis == "z":
        nonroll.attr("ro").v = mrc.Ro.zxy

    e2q, q2etw, q2esw, quatinv, quatprod = swing_twist_nodes(axis)

    nonroll.attr("ro") >> e2q.attr("inputRotateOrder")
    nonroll.attr("ro") >> q2etw.attr("inputRotateOrder")
    nonroll.attr("ro") >> q2esw.attr("inputRotateOrder")

    oricons.attr("cr") >> e2q.attr("inputRotate")

    # Twist
    nonroll.add(ln="twist", k=True)
    outrot = "outputRotate{}".format(axis.upper())
    q2etw.attr(outrot) >> nonroll.attr("twist")

    # Swing
    q2esw.attr("outputRotateX") >> nonroll.attr("rx")
    q2esw.attr("outputRotateY") >> nonroll.attr("ry")
    q2esw.attr("outputRotateZ") >> nonroll.attr("rz")

    return nonroll, pointcons, oricons, e2q, q2etw, q2esw, quatinv, quatprod


def extract_swing_twist(
    dag: mrc.Dag, axis: str
) -> Tuple[mrc.Node, mrc.Node, mrc.Node, mrc.Node, mrc.Node]:
    """Extracts swing and twist rotations from the given dag.

    Returns eulerToQuat, quatToEuler(twist), quatToEuler(swing), quatInvert
    and quatProd nodes.
    """
    dag.add(ln="swing", at="double3")
    dag.add(ln="swingX", at="double", p="swing")
    dag.add(ln="swingY", at="double", p="swing")
    dag.add(ln="swingZ", at="double", p="swing")
    dag.add(ln="twist")

    e2q, q2etw, q2esw, quatinv, quatprod = swing_twist_nodes(axis)

    dag.attr("r") >> e2q.attr("inputRotate")
    dag.attr("ro") >> e2q.attr("inputRotateOrder")
    dag.attr("ro") >> q2etw.attr("inputRotateOrder")
    dag.attr("ro") >> q2esw.attr("inputRotateOrder")

    # Twist
    outrot = "outputRotate{}".format(axis.upper())
    q2etw.attr(outrot) >> dag.attr("twist")

    # Swing
    q2esw.attr("outputRotate") >> dag.attr("swing")

    return e2q, q2etw, q2esw, quatinv, quatprod


def add_divide_attr(ctrl: mrc.Dag, attrname: str) -> mrc.Attr:
    """Adds divide attribute to be used as a divider in channel box.
    It starts and ends with '__' and is locked.
    """
    attrname = "__{}__".format(attrname.strip("_"))
    ctrl.add(ln=attrname, k=True)
    ctrl.attr(attrname).lock = True

    return ctrl.attr(attrname)


def add_shape_vis_controller(ctrl: mrc.Dag, target: mrc.Dag, attrname: str):
    """Adds the given 'attrname' to ctrl.shape to control the visibility of
    the shape node of the given target, if the given ctrl doesn't have
    a shape, adds to itself instead.
    """
    ctrlshape = ctrl.get_shape()
    if ctrlshape:
        if not ctrlshape.attr(attrname).exists:
            ctrlshape.add(ln=attrname, k=True, min=0, max=1)
        driver = ctrlshape.attr(attrname)
    else:
        if not ctrl.attr(attrname).exists:
            ctrl.add(ln=attrname, k=True, min=0, max=1)
        driver = ctrl.attr(attrname)

    if target.get_shape():
        target = target.get_shape()

    driver >> target.attr("v")


def m_matrix_to_tuple(mat: om.MMatrix) -> Tuple[tuple, tuple, tuple, tuple]:
    """Translates om.MMatrix to a tuple.
    Ex.
        (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    """
    return tuple(
        tuple(om.MScriptUtil.getDoubleArrayItem(mat[r], c) for c in range(4))
        for r in range(4)
    )


def rotate_between_two_objects(
    dag: mrc.Dag, start_dag: mrc.Dag, end_dag: mrc.Dag, weight: float
):
    """Sets rotation of the given dag to weighted rotate between
    start_obj and end_obj with the given weight.
    """
    start_obj_mat = start_dag.m_dag_path.inclusiveMatrix()
    start_obj_xform_mat = om.MTransformationMatrix(start_obj_mat)
    start_obj_quat = start_obj_xform_mat.rotation()

    end_obj_mat = end_dag.m_dag_path.inclusiveMatrix()
    end_obj_xform_mat = om.MTransformationMatrix(end_obj_mat)
    end_obj_quat = end_obj_xform_mat.rotation()

    halfway_quat = slerp(start_obj_quat, end_obj_quat, weight)

    xformfn = om.MFnTransform(dag.m_dag_path)
    eulerrot = halfway_quat.asEulerRotation()
    eulerrot.reorderIt(xformfn.rotationOrder() - 1)
    degrees = [
        math.degrees(angle) for angle in [eulerrot.x, eulerrot.y, eulerrot.z]
    ]

    dag.attr("r").v = degrees


def slerp(
    quat_a: om.MQuaternion, quat_b: om.MQuaternion, weight: float
) -> om.MQuaternion:
    """Finds slerp between the given quat_a and quat_b with the given weight.
    The original code is from
    https://github.com/chadmv/cmt/blob/master/scripts/cmt/rig/swingtwist.py
    """
    result = om.MQuaternion()

    # Calculate angle between them.
    cos_half_theta = (
        quat_a.w * quat_b.w
        + quat_a.x * quat_b.x
        + quat_a.y * quat_b.y
        + quat_a.z * quat_b.z
    )
    if abs(cos_half_theta) >= 1.0:
        result.w = quat_a.w
        result.x = quat_a.x
        result.y = quat_a.y
        result.z = quat_a.z
        return quat_a

    # Calculate temporary values
    half_theta = math.acos(cos_half_theta)
    sin_half_theta = math.sqrt(1.0 - cos_half_theta * cos_half_theta)
    # if theta = 180 degrees then result is not fully defined
    # we could rotate around any axis normal to quat_a or quat_b
    if math.fabs(sin_half_theta) < 0.001:
        result.w = quat_a.w * 0.5 + quat_b.w * 0.5
        result.x = quat_a.x * 0.5 + quat_b.x * 0.5
        result.y = quat_a.y * 0.5 + quat_b.y * 0.5
        result.z = quat_a.z * 0.5 + quat_b.z * 0.5
        return result

    ratio_a = math.sin((1 - weight) * half_theta) / sin_half_theta
    ratio_b = math.sin(weight * half_theta) / sin_half_theta

    # Calculate quaternion
    result.w = quat_a.w * ratio_a + quat_b.w * ratio_b
    result.x = quat_a.x * ratio_a + quat_b.x * ratio_b
    result.y = quat_a.y * ratio_a + quat_b.y * ratio_b
    result.z = quat_a.z * ratio_a + quat_b.z * ratio_b

    return result


def add_space_blender(
    target_nodes_and_attr_names: Tuple[Tuple[mrc.Dag, str]],
    cons_func: Union[mrc.pntcons, mrc.oricons, mrc.parcons],
    constrained_node: mrc.Dag,
    controller: mrc.Dag,
) -> Tuple[List[mrc.Dag], List[mrc.Node], List[str]]:
    """Adds space blender with the given cons_func to the constrained_node.

    Args:
        target_nodes_and_attr_names (tuple): A tuple of tuples that have
            first items as target nodes and scond item as attribute names.
            Ex. ((prc.Dag('Head_Jnt'): 'head'), (prc.Dag('Eye_Jnt'): 'eye'))
        cons_func (prc.orient_constraint/prc.point_constraint/
            prc.parent_constraint):
        constrained_node (prc.Dag): 'Local' space is always added as the first
            space blender.
            It uses constrained_node's parent as its space.
        controller (prc.Dag):

    Returns:
        list: A list of prc.Dag, space nodes.
        list: A list of prc.Node, condition nodes.
        list: A list of str, attribute names.
    """
    # Defines attribute prefix.
    attr_prefix = "pointTo"
    if cons_func == mrc.oricons:
        attr_prefix = "orientTo"
    elif cons_func == mrc.parcons:
        attr_prefix = "parentTo"

    # Collects all target nodes.
    # The local space is added as the first space.
    parent = constrained_node.get_parent()

    targets = [parent] + [i[0] for i in target_nodes_and_attr_names]
    attrs = ["local"] + [i[1] for i in target_nodes_and_attr_names]

    # Adds enum attribute.
    controller.add(ln=attr_prefix, at="enum", en=":".join(attrs), k=True)

    # Creates target nodes.
    space_nodes = []
    for attr, tg in zip(attrs, targets):
        spc = mrc.Null()
        spc.name = attr
        space_nodes.append(spc)

        spc.snap(constrained_node)
        spc.set_parent(tg)

    cons = cons_func(*space_nodes + [constrained_node], mo=True)

    conds = []
    for ix in range(len(space_nodes)):
        cond = mrc.create("condition")
        conds.append(cond)

        controller.attr(attr_prefix) >> cond.attr("firstTerm")
        cond.attr("secondTerm").v = ix
        cond.attr("colorIfTrueR").v = 1
        cond.attr("colorIfFalseR").v = 0
        cond.attr("outColorR") >> cons.attr("w{}".format(ix))

    return space_nodes, conds, attrs


def do_positions_match(dag1: mrc.Dag, dag2: mrc.Dag, tol=0.001) -> bool:
    """Compares the positions of the given dag1 and dag2.
    Returns True if the positions match with the given tolerance.
    """
    dag1pos = dag1.world_pos
    dag2pos = dag2.world_pos
    for ix in range(3):
        if abs(dag2pos[ix] - dag1pos[ix]) > tol:
            return False
    return True


def do_vectors_match(vec1: list, vec2: list, tol=0.001) -> bool:
    """Compares two vectors of three components.
    Returns True if all values are match with the given tolerance.
    """
    for ix in range(3):
        if abs(vec2[ix] - vec1[ix]) > tol:
            return False

    return True


def get_curve_lentgh(curve: mrc.Dag) -> float:
    curvedag = mrc.dag_path_from_dag_str(curve.name)
    curvedag.extendToShape()
    curvefn = om.MFnNurbsCurve(curvedag)

    return curvefn.length()


def get_param_of_closest_point_on_curve(curve: mrc.Dag, dag: mrc.Dag) -> float:
    """Gets curve paramter of the given curve that has the closest
    distant to the given dag.
    """
    curvedag = mrc.dag_path_from_dag_str(curve.name)

    curvedag.extendToShape()
    curvefn = om.MFnNurbsCurve(curvedag)

    point = om.MPoint(*dag.world_pos)
    paramutil = om.MScriptUtil()
    paramptr = paramutil.asDoublePtr()

    curvefn.closestPoint(point, paramptr)

    return paramutil.getDouble(paramptr)


def get_param_of_closest_point_on_surface(
    nurb: mrc.Dag, dag: mrc.Dag
) -> Tuple[float, float]:
    """Gets surface paramters of the given nurb that has the closest
    distant to the given dag.

    Returns U and V parameters as floats.
    """
    nurbdag = mrc.dag_path_from_dag_str(nurb.name)

    nurbdag.extendToShape()
    nurbfn = om.MFnNurbsSurface(nurbdag)

    point = om.MPoint(*dag.world_pos)

    closest_point = nurbfn.closestPoint(point)

    uparamutil = om.MScriptUtil()
    vparamutil = om.MScriptUtil()

    uptr = uparamutil.asDoublePtr()
    vptr = vparamutil.asDoublePtr()
    nurbfn.getParamAtPoint(closest_point, uptr, vptr)

    return float(uparamutil.getDouble(uptr)), float(vparamutil.getDouble(vptr))


def get_point_at_surface_param(
    nurb: mrc.Dag, u: float, v: float
) -> Tuple[float, float, float]:
    """Gets world position at the given surface parameters.

    Returns a tuple of x, y, z values.
    """
    nurbdag = mrc.dag_path_from_dag_str(nurb.name)

    nurbdag.extendToShape()
    nurbfn = om.MFnNurbsSurface(nurbdag)

    p = om.MPoint()
    nurbfn.getPointAtParam(u, v, p)

    return p.x, p.y, p.z


def get_point_at_curve_param(
    curve: mrc.Dag, p: float
) -> Tuple[float, float, float]:
    """Gets world position at the given curve parameters.

    Returns a tuple of x, y, z values.
    """
    curvedag = mrc.dag_path_from_dag_str(curve.name)

    curvedag.extendToShape()
    curvefn = om.MFnNurbsCurve(curvedag)

    point = om.MPoint()
    curvefn.getPointAtParam(p, point, om.MSpace.kWorld)

    return point.x, point.y, point.z


def get_normal_at_curve_param(
    curve: mrc.Dag, p: float
) -> Tuple[float, float, float]:
    """Gets normal vector at the given curve parameters."""
    curvedag = mrc.dag_path_from_dag_str(curve.name)

    curvedag.extendToShape()
    curvefn = om.MFnNurbsCurve(curvedag)

    n = curvefn.normal(p, om.MSpace.kWorld)

    return n.x, n.y, n.z


def get_tangent_at_curve_param(
    curve: mrc.Dag, p: float
) -> Tuple[float, float, float]:
    """Gets tangent vector at the given curve parameters."""
    curvedag = mrc.dag_path_from_dag_str(curve.name)

    curvedag.extendToShape()
    curvefn = om.MFnNurbsCurve(curvedag)

    t = curvefn.tangent(p, om.MSpace.kWorld)

    return t.x, t.y, t.z


def get_closest_mesh_uv(
    worldpos: Tuple[float, float, float], geo: mrc.Dag
) -> Tuple[float, float]:
    """Finds the closest UV coordinates to a given world position
    on a specified mesh.
    """
    point = om.MPoint(*worldpos)

    dagpath = geo.get_shape().m_dag_path
    meshfn = om.MFnMesh(dagpath)
    msutil = om.MScriptUtil()
    msutil.createFromDouble(0.0, 0.0)
    uvptr = msutil.asFloat2Ptr()
    meshfn.getUVAtPoint(point, uvptr, om.MSpace.kWorld, "map1", None)
    u = msutil.getFloat2ArrayItem(uvptr, 0, 0)
    v = msutil.getFloat2ArrayItem(uvptr, 0, 1)

    return u, v


def create_pos(
    nurb: mrc.Dag, dag: mrc.Dag, normal_to: int, u_to: int, v_to: int
) -> Tuple[mrc.Node, mrc.Node, mrc.Node, mrc.Node]:
    """Uses fourByFourMatrix, multMatrix and decomposeMatrix nodes to
    get a rivet likes rig to the given DAG node.

    Args:
        nurb (prc.Dag): A transform node of the NURBs surface.
        dag (prc.Dag): As a rivet node.
        normal_to (int): 0, 1 or 2 as a row of Matrix that normalized normal
            connects to.
        u_to (int): 0, 1 or 2 as a row of Matrix that normalized tangent U
            connects to.
        v_to (int): 0, 1 or 2 as a row of Matrix that normalized tangent V
            connects to.

    Returns:
        prc.Node: pointOnSurfaceInfo
        prc.Node: fourByFourMatrix
        prc.Node: multMatrix
        prc.Node: decomposeMatrix
    """
    shape = nurb.get_shape()

    posi = mrc.create("pointOnSurfaceInfo")
    fbf = mrc.create("fourByFourMatrix")

    shape.attr("worldSpace[0]") >> posi.attr("inputSurface")
    posi.attr("positionX") >> fbf.attr("in30")
    posi.attr("positionY") >> fbf.attr("in31")
    posi.attr("positionZ") >> fbf.attr("in32")

    posi.attr("normalizedNormalX") >> fbf.attr("in{}0".format(normal_to))
    posi.attr("normalizedNormalY") >> fbf.attr("in{}1".format(normal_to))
    posi.attr("normalizedNormalZ") >> fbf.attr("in{}2".format(normal_to))

    posi.attr("normalizedTangentUX") >> fbf.attr("in{}0".format(u_to))
    posi.attr("normalizedTangentUY") >> fbf.attr("in{}1".format(u_to))
    posi.attr("normalizedTangentUZ") >> fbf.attr("in{}2".format(u_to))

    posi.attr("normalizedTangentVX") >> fbf.attr("in{}0".format(v_to))
    posi.attr("normalizedTangentVY") >> fbf.attr("in{}1".format(v_to))
    posi.attr("normalizedTangentVZ") >> fbf.attr("in{}2".format(v_to))

    multmat, decomp = mult_by_parent_inverse_mat(
        dag, fbf.attr("output"), True, True, False
    )

    return posi, fbf, multmat, decomp


def meuler_rotation_to_tuple(
    eul: om.MEulerRotation,
) -> Tuple[float, float, float]:
    """Converts MEulerRotation to a tuple of three floats."""
    return (math.degrees(eul.x), math.degrees(eul.y), math.degrees(eul.z))


def aim(
    dag: mrc.Dag,
    target: mrc.Dag,
    dag_aim: Tuple[float, float, float],
    dag_up: Tuple[float, float, float],
    up_vec: Tuple[float, float, float],
):
    """Orients dag_aim axis of dag to target node and aligns dag_up axis
    to up_vec.
    """
    mc.delete(
        mrc.aim_constraint(
            target, dag, aim=dag_aim, u=dag_up, wu=up_vec, wut="vector"
        )
    )


def add_two_ways_clamp(
    driver: mrc.Attr, minr: float, maxr: float, ming: float, maxg: float
) -> mrc.Node:
    """Clamps the driver, using min/max values, to two channels."""
    clamp = mrc.create("clamp")
    clamp.attr("minR").v = minr
    clamp.attr("maxR").v = maxr
    clamp.attr("minG").v = ming
    clamp.attr("maxG").v = maxg

    driver >> clamp.attr("inputR")
    driver >> clamp.attr("inputG")

    return clamp


def attr_clamper(
    driver: mrc.Attr,
    driven: Union[mrc.Attr, None],
    min_value: float,
    max_value: float,
) -> mrc.Node:
    """Adds a clamp node to clamp value of the driver regarding the given
    min/max_value.
    The output from clamp node connects to the driven if it's given.
    """
    clamper = mrc.create("clamp")

    clamper.attr("minR").v = min_value
    clamper.attr("maxR").v = max_value

    driver >> clamper.attr("inputR")
    if driven:
        clamper.attr("outputR") >> driven

    return clamper


def attr_amper(driver: mrc.Attr, driven: mrc.Attr, dv: float) -> mrc.Node:
    """Adds an attribute amplifier to the driver that drives
    the driven attribute. If the driver node has shape the amp_attr is added
    to its shape else is added to the driver node.

    Returns a multDoubleLinear node.
    """
    mult = mrc.create("multDoubleLinear")
    driver >> mult.attr("i1")
    mult.attr("i2").v = dv
    mult.attr("o") >> driven

    return mult


def create_joint_at(target: Union[mrc.Node, mrc.Component]) -> mrc.Joint:
    """Creates a joint at the position of the given transform."""
    jnt = mrc.Joint()
    if mrc.Dag in type(target).__mro__:
        jnt.snap(target)
        jnt.freeze()
        jnt.attr("rotateOrder").v = target.attr("rotateOrder").v
    elif mrc.Component in type(target).__mro__:
        jnt.world_pos = target.world_pos

    return jnt


def create_joint_and_its_zero_at(dag: mrc.Dag) -> Tuple[mrc.Joint, mrc.Dag]:
    """Creates a joint and its zero group at the position of
    the given transform.
    """
    jnt = mrc.Joint()
    zr = mrc.group(jnt)
    zr.snap(dag)

    return jnt, zr


def add_zero_group(dag: mrc.Dag) -> mrc.Dag:
    parent = dag.get_parent()

    zr = mrc.Null()
    zr.snap(dag)
    dag.set_parent(zr)

    if parent:
        zr.set_parent(parent)

    return zr


def dup_and_clean_unused_intermediate_shape(xform: mrc.Dag) -> mrc.Dag:
    dupped = mrc.Dag(mc.duplicate(xform, rr=True)[0])
    shapes = dupped.get_all_shapes()

    for attr in ("t", "r", "s", "v"):
        dupped.attr(attr).lock = False
        dupped.attr(attr).hide = False

    if not shapes:
        return dupped

    for shape in shapes:
        if shape.attr("intermediateObject").v:
            mc.delete(shape)

    return dupped


def tr_blend(
    dag1: mrc.Dag, dag2: mrc.Dag, blended_dag: mrc.Dag, driver: mrc.Attr
) -> mrc.Node:
    """Adds translate and rotate blend using a pairBlend node."""
    blend = mrc.create("pairBlend")
    blend.attr("rotInterpolation").v = 0

    driver >> blend.attr("weight")
    dag1.attr("t") >> blend.attr("inTranslate1")
    dag1.attr("r") >> blend.attr("inRotate1")
    dag2.attr("t") >> blend.attr("inTranslate2")
    dag2.attr("r") >> blend.attr("inRotate2")

    blend.attr("outTranslate") >> blended_dag.attr("t")
    blend.attr("outRotate") >> blended_dag.attr("r")

    return blend


def s_blend(
    dag1: mrc.Dag, dag2: mrc.Dag, blended_dag: mrc.Dag, driver: mrc.Attr
) -> mrc.Node:
    """Adds scale blend using a blendColors node."""
    blend = mrc.create("blendColors")

    driver >> blend.attr("blender")
    dag1.attr("s") >> blend.attr("color1")
    dag2.attr("s") >> blend.attr("color2")
    blend.attr("output") >> blended_dag.attr("s")

    return blend


def get_compo_center(compoes: List[mrc.Component]) -> om.MVector:
    """Finds center of the given components."""
    center = [0, 0, 0]
    for compo in compoes:
        pos = compo.world_pos
        center[0] += pos[0]
        center[1] += pos[1]
        center[2] += pos[2]
    return om.MVector(
        center[0] / len(compoes),
        center[1] / len(compoes),
        center[2] / len(compoes),
    )


def get_bounding_box_under(
    dag: mrc.Dag, compoes: List[mrc.Component]
) -> Tuple[float, float, float, float, float]:
    """Gets bounding box dimension of the given components
    under the given dag's space.
    """
    origin = get_compo_center(mrc.to_components(mrc.get_all_components(dag)))

    x_vals = []
    y_vals = []
    z_vals = []
    for compo in compoes:
        vec = om.MVector(*compo.world_pos) - origin
        x_vals.append(vec * om.MVector(*dag.x_axis))
        y_vals.append(vec * om.MVector(*dag.y_axis))
        z_vals.append(vec * om.MVector(*dag.z_axis))

    return (
        min(x_vals),
        min(y_vals),
        min(z_vals),
        max(x_vals),
        max(y_vals),
        max(z_vals),
    )


def auto_width_controller(
    dag: mrc.Dag, compoes: List[mrc.Component], scale: float
) -> None:
    """Scales and moves the given dag object to match with
    the given compo objects.
    """
    dag_compoes = mrc.to_components(mrc.get_all_components(dag))

    compo_center = get_compo_center(compoes)
    dag_center = get_compo_center(dag_compoes)

    move_vec = compo_center - dag_center
    mc.move(
        move_vec.x,
        move_vec.y,
        move_vec.z,
        mrc.get_all_components(dag.name),
        r=True,
    )
    dag_center = get_compo_center(dag_compoes)

    compo_bb = get_bounding_box_under(dag, compoes)
    dag_bb = get_bounding_box_under(dag, dag_compoes)

    dag_x = dag_bb[3] - dag_bb[0]
    dag_z = dag_bb[5] - dag_bb[2]

    compo_x = compo_bb[3] - compo_bb[0]
    compo_z = compo_bb[5] - compo_bb[2]

    x_scale = (compo_x / dag_x) * scale
    z_scale = (compo_z / dag_z) * scale

    mc.scale(
        x_scale,
        1,
        z_scale,
        mrc.get_all_components(dag),
        pivot=(dag_center.x, dag_center.y, dag_center.z),
        r=True,
    )


def auto_width_selected_controller(scale: float) -> None:
    """Select components, a controller then call this function."""
    sels = mc.ls(sl=True, fl=True)
    dag = mrc.Dag(sels[-1])
    compoes = mrc.to_components(sels[:-1])
    auto_width_controller(dag, compoes, scale)


def connect_shape_vis(driver: mrc.Attr, dag: mrc.Dag) -> None:
    """Connects the given driver attribute to the shapes of the given dag."""
    for shape in dag.get_all_shapes():
        olddriver = shape.attr("v").get_input()
        if olddriver:
            olddriver // shape.attr("v")
        driver >> shape.attr("v")


def connect_shape_vises(driver: mrc.Attr, dags: List[mrc.Dag]) -> None:
    """Connects the given driver attribute to the shapes of the given dags."""
    for dag in dags:
        connect_shape_vis(driver, dag)


def selected_edge_to_control_shape(closed_curve: bool, cv_amount: int) -> None:
    """Select edges then a transform node.

    Script converts selected edges to NURBs curve.
    If "Closed Curve" is checked, script generates the NURBs circle and matches
    the created curve with the selected edges.
    """
    dag = mrc.Dag(mc.ls(sl=True)[-1])
    mc.select(dag, d=True)
    guide_crv = mrc.Dag(mc.polyToCurve(form=0, degree=1, ch=False)[0])
    spans = guide_crv.get_shape().attr("spans").v
    cv_step = float(spans) / float(cv_amount)

    pnts = [
        get_point_at_curve_param(guide_crv, ix * cv_step)
        for ix in range(cv_amount)
    ]

    if closed_curve:
        crv = mrc.Dag(mc.circle(d=3, s=len(pnts) - 1, ch=False)[0])
    else:
        pnts.append(pnts[0])
        crv = mrc.Dag(mc.curve(d=1, p=[(0, 0, 0)] * len(pnts)))

    shape = crv.get_shape()
    mc.parent(shape, dag, s=True, r=True)
    for ix, pnt in enumerate(pnts):
        mc.xform("{}.cv[{}]".format(shape, ix), t=pnt, ws=True)
    mc.delete(guide_crv, crv)


def selected_edge_to_control_shape_ui() -> None:
    """A UI for selected_edge_to_control_shape"""

    def _button_triggered(*_):
        cv_amt = mc.intFieldGrp(cv_amt_ffg, q=True, v1=True)
        cc = mc.checkBox(cc_cb, q=True, v=True)
        selected_edge_to_control_shape(cc, cv_amt)

    ui = "selected_edges_to_curve"
    win = "{}_win".format(ui)

    if mc.window(win, exists=True):
        mc.deleteUI(win)

    mc.window(win, t="Seclect mesh edges")
    mc.columnLayout(adj=True)
    cv_amt_ffg = mc.intFieldGrp(
        nf=1, l="CV amount:", v1=10, cal=[1, "left"], cw2=[80, 140]
    )
    cc_cb = mc.checkBox(label="Closed Curve", v=True)
    mc.button(l="Create Curve", c=partial(_button_triggered), h=50)

    mc.showWindow(win)
    mc.window(win, e=True, w=220)
    mc.window(win, e=True, h=115)


def get_first_connected_skin_cluster(name: str):
    hists = mc.listHistory(name)
    if not hists:
        return False
    for his in hists:
        if mc.nodeType(his) == "skinCluster":
            return his


def copy_weight():
    """Copy skin weights from first selected object to the rest selected object(s).
    Select source geometry then target geometries then run script.
    """
    sels = mc.ls(sl=True)
    jnts = mc.skinCluster(sels[0], q=True, inf=True)

    for sel in sels[1:]:
        oskin = get_first_connected_skin_cluster(sel)
        if oskin:
            mc.skinCluster(oskin, e=True, ub=True)

        mc.skinCluster(jnts, sel, tsb=True)[0]

        mc.select((sels[0], sel), r=True)
        mc.select(sel, add=True)
        mc.copySkinWeights(
            noMirror=True,
            surfaceAssociation="closestPoint",
            influenceAssociation="closestJoint",
        )
