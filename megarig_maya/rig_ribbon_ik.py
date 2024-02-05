from typing import Union, Tuple

import maya.OpenMaya as om
import maya.cmds as mc

from megarig_maya import core as mrc
from megarig_maya import rig_base
from megarig_maya import utils as mru


class RibbonCurve(rig_base.BaseRig):
    """Generates tweaker rigs along the given curve
    with auto twist and auto squash functions.
    Tweaker is aligned to y axis, z axis is oriented to up curve.
    """

    def __init__(
        self,
        crv: mrc.Dag,
        up_crv: mrc.Dag,
        jnt_amt: int,
        mod_name: str,
        desc: str,
        side: mrc.Side,
        ctrl_shp: mrc.Cp,
        ctrl_clr: mrc.Color,
        parent: Union[mrc.Dag, None],
        skin_parent: Union[mrc.Dag, None],
    ):
        super(RibbonCurve, self).__init__(mod_name, desc, side)

        if parent:
            self.meta.set_parent(parent)

        self.jnt_amt = jnt_amt
        self.skin_parent = skin_parent
        self.crv = crv
        self.up_crv = up_crv

        self.crv_shp = self.crv.get_shape()
        self.up_crv_shp = self.up_crv.get_shape()

        self.still_grp = self.init_dag(
            mrc.Null(), self.side, ("still",), "grp"
        )

        # Arc Lengthes
        mru.add_divide_attr(self.meta, "length")
        self.cif = self.init_node(
            mrc.create("curveInfo"), self.side, ("",), "cif"
        )
        self.crv_shp.attr("ws[0]") >> self.cif.attr("ic")
        self.curr_len = self.cif.attr("al")
        self.orig_len = self.meta.add(ln="origLen", k=True, dv=self.curr_len.v)

        # Twist Attrs
        mru.add_divide_attr(self.meta, "twist")
        self.tw = self.meta.add(ln="twist", k=True)
        self.root_tw = self.meta.add(ln="rootTwist", k=True)
        self.end_tw = self.meta.add(ln="endTwist", k=True)

        # Squash Attrs
        mru.add_divide_attr(self.meta, "squash")
        self.auto_squash = self.meta.add(ln="autoSquash", k=True, min=0, max=1)
        self.squash = self.meta.add(ln="squash", k=True)
        self.min_sc = self.meta.add(ln="mnSc", dv=0.4, k=True)
        self.max_sc = self.meta.add(ln="mxSc", dv=2, k=True)
        self.squash_amps = []
        for ix in range(self.jnt_amt):
            self.squash_amps.append(
                self.meta.add(ln="squashAmp{}".format(ix + 1), k=True, dv=1)
            )

        # Squesh
        self.sq_len_clamp = self.init_node(
            mrc.create("clamp"), self.side, ("sq",), "clamp"
        )

        def _add_sq_len_mnmx(name: str, driver: mrc.Attr) -> mrc.Attr:
            mult = self.init_node(
                mrc.create("multDoubleLinear"), self.side, ("sq", name), "mult"
            )
            self.orig_len >> mult.attr("i2")
            driver >> mult.attr("i1")

            return mult.attr("o")

        self.sq_len_mn = _add_sq_len_mnmx("mn", self.min_sc)
        self.sq_len_mx = _add_sq_len_mnmx("mx", self.max_sc)

        self.curr_len >> self.sq_len_clamp.attr("ipr")
        self.sq_len_mn >> self.sq_len_clamp.attr("mnr")
        self.sq_len_mx >> self.sq_len_clamp.attr("mxr")

        self.sq_div = self.init_node(
            mrc.create("multiplyDivide"), self.side, ("sq",), "div"
        )
        self.sq_pow = self.init_node(
            mrc.create("multiplyDivide"), self.side, ("sq",), "pow"
        )
        self.sq_curr_len_div = self.init_node(
            mrc.create("multiplyDivide"),
            self.side,
            ("sq", "curr", "len"),
            "div",
        )

        self.sq_div.attr("op").v = mrc.Op.divide
        self.sq_pow.attr("op").v = mrc.Op.power
        self.sq_curr_len_div.attr("op").v = mrc.Op.divide

        self.sq_len_clamp.attr("opr") >> self.sq_curr_len_div.attr("i1x")
        self.orig_len >> self.sq_curr_len_div.attr("i2x")

        self.sq_curr_len_div.attr("ox") >> self.sq_pow.attr("i1x")
        self.sq_pow.attr("i2x").v = 2
        self.sq_div.attr("i1x").v = 1
        self.sq_pow.attr("ox") >> self.sq_div.attr("i2x")

        self.sq_sub_one = self.init_node(
            mrc.create("addDoubleLinear"),
            self.side,
            ("sq", "one"),
            "sub",
        )
        self.sq_sub_one.attr("i1").v = -1
        self.sq_div.attr("ox") >> self.sq_sub_one.attr("i2")

        self.sq_out = self.sq_sub_one.attr("o")

        # Tweakers
        self.ctrls = []
        self.ctrl_ofsts = []
        self.ctrl_ories = []
        self.ctrl_zrs = []
        self.mps = []
        self.up_mps = []
        self.sqs = []

        def _add_rig(index: int):
            idstr = str(index)
            (
                ctrl,
                ofst,
                ori,
                zr,
            ) = self.add_controller(
                self.side, (idstr,), ctrl_shp, ctrl_clr, ("v",)
            )
            ctrl.scale_shape(self.orig_len.v / 10)
            zr.set_parent(self.meta)

            def _add_mp(shp: mrc.Dag, names: Tuple[str]) -> mrc.Node:
                mp = self.init_node(
                    mrc.create("motionPath"), self.side, names, "mp"
                )
                shp.attr("ws[0]") >> mp.attr("gp")

                return mp

            mp = _add_mp(self.crv_shp, (idstr,))
            ump = _add_mp(self.up_crv_shp, ("up", idstr))

            uobj = self.init_dag(mrc.Null(), self.side, ("up", idstr), "grp")
            uobj.set_parent(self.still_grp)

            ump.attr("ac") >> uobj.attr("t")
            uobj.attr("wm[0]") >> mp.attr("wum")
            mp.attr("ac") >> zr.attr("t")
            mp.attr("wut").v = 1
            mp.attr("fa").v = 1
            mp.attr("ua").v = 2

            param = (1.0 / float(self.jnt_amt - 1)) * float(index - 1)
            mp.attr("u").v = param
            ump.attr("u").v = param

            # Twist
            tw_add = self.init_node(
                mrc.create("addDoubleLinear"), self.side, ("tw", idstr), "add"
            )
            tw_sum = self.init_node(
                mrc.create("addDoubleLinear"),
                self.side,
                ("tw", "aum", idstr),
                "add",
            )
            tw_root = self.init_node(
                mrc.create("multDoubleLinear"),
                self.side,
                ("tw", idstr),
                "mult",
            )
            tw_end = self.init_node(
                mrc.create("multDoubleLinear"),
                self.side,
                ("tw", "end", idstr),
                "mult",
            )

            self.root_tw >> tw_root.attr("i1")
            tw_root.attr("i2").v = 1.0 - param
            self.end_tw >> tw_end.attr("i1")
            tw_end.attr("i2").v = param
            tw_root.attr("o") >> tw_sum.attr("i1")
            tw_end.attr("o") >> tw_sum.attr("i2")
            self.tw >> tw_add.attr("i1")
            tw_sum.attr("o") >> tw_add.attr("i2")

            # Twist - Quat Product
            mp_e2q = self.init_node(
                mrc.create("eulerToQuat"), self.side, ("mp", idstr), "e2q"
            )
            tw_e2q = self.init_node(
                mrc.create("eulerToQuat"), self.side, ("tw", idstr), "e2q"
            )
            quat_prod = self.init_node(
                mrc.create("quatProd"), self.side, ("tw", idstr), "quatProd"
            )

            mp.attr("r") >> mp_e2q.attr("inputRotate")
            tw_add.attr("o") >> tw_e2q.attr("iry")
            tw_e2q.attr("oq") >> quat_prod.attr("iq1")
            mp_e2q.attr("oq") >> quat_prod.attr("iq2")

            # Squesh
            sq_blend = self.init_node(
                mrc.create("blendTwoAttr"), self.side, ("sq", idstr), "blend"
            )
            sq_amp = self.init_node(
                mrc.create("multDoubleLinear"),
                self.side,
                ("sq", idstr),
                "mult",
            )
            sq_add_one = self.init_node(
                mrc.create("addDoubleLinear"),
                self.side,
                ("sq", "one", idstr),
                "add",
            )
            sq_add = self.init_node(
                mrc.create("addDoubleLinear"),
                self.side,
                ("sq", idstr),
                "add",
            )
            self.auto_squash >> sq_blend.attr("ab")
            self.sq_out >> sq_amp.attr("i1")
            self.squash_amps[index - 1] >> sq_amp.attr("i2")
            sq_add_one.attr("i1").v = 1
            sq_amp.attr("o") >> sq_add_one.attr("i2")

            sq_blend.attr("i[0]").v = 1
            sq_add_one.attr("o") >> sq_blend.attr("i[1]")

            sq_blend.attr("o") >> sq_add.attr("i1")
            self.squash >> sq_add.attr("i2")

            # Xform Matrices
            comp_mat = self.init_node(
                mrc.create("composeMatrix"), self.side, (idstr,), "compMat"
            )
            decomp_mat = self.init_node(
                mrc.create("decomposeMatrix"), self.side, (idstr,), "decompMat"
            )

            comp_mat.attr("uer").v = False
            quat_prod.attr("oq") >> comp_mat.attr("iq")
            mp.attr("ac") >> comp_mat.attr("it")
            comp_mat.attr("omat") >> decomp_mat.attr("imat")
            zr.attr("ro") >> decomp_mat.attr("ro")
            decomp_mat.attr("ot") >> zr.attr("t")
            decomp_mat.attr("or") >> zr.attr("r")

            self.ctrls.append(ctrl)
            self.ctrl_ofsts.append(ofst)
            self.ctrl_ories.append(ori)
            self.ctrl_zrs.append(zr)
            self.mps.append(mp)
            self.up_mps.append(ump)
            self.sqs.append(sq_add.attr("o"))

        for ix in range(self.jnt_amt):
            _add_rig(ix + 1)

        # Joints
        self.master_jnt = self.init_dag(
            mrc.Joint(), self.side, ("master",), "jnt"
        )
        self.master_jnt.attr("ds").v = mrc.DrawStyle.none
        if self.skin_parent:
            self.master_jnt.snap(self.skin_parent)
            self.master_jnt.freeze()

        self.jnts = []
        for ix in range(self.jnt_amt):
            jnt = self.init_dag(
                mru.create_joint_at(self.ctrls[ix]),
                self.side,
                (str(ix + 1),),
                "jnt",
            )
            self.jnts.append(jnt)
            jnt.attr("radi").v = self.orig_len.v / 10
            jnt.set_parent(self.master_jnt)

            jnt.freeze()
            cons = mrc.parent_constraint(self.ctrls[ix], jnt)
            cons.set_parent(self.still_grp)

        # Connect scale
        for ix in range(self.jnt_amt):
            sc_sum = self.init_node(
                mrc.create("plusMinusAverage"),
                self.side,
                (
                    "sc",
                    str(ix + 1),
                ),
                "sum",
            )
            sc_sum.attr("i3[0]").v = (-1, 0, -1)
            self.sqs[ix] >> sc_sum.attr("i3[1].i3x")
            self.sqs[ix] >> sc_sum.attr("i3[1].i3z")
            self.ctrls[ix].attr("s") >> sc_sum.attr("i3[2]")
            sc_sum.attr("o3") >> self.jnts[ix].attr("s")


class RibbonIk(rig_base.BaseRig):
    def __init__(
        self,
        root: mrc.Dag,
        end: mrc.Dag,
        jnt_amt: int,
        side: mrc.Side,
        mod_name: str,
        desc: str,
        ctrl_shp: mrc.Cp,
        dtl_ctrl_shp: mrc.Cp,
        ctrl_clr: mrc.Color,
        dtl_clr: mrc.Color,
        parent: Union[mrc.Dag, None],
        skin_parent: Union[mrc.Dag, None],
        still_parent: Union[mrc.Dag, None],
    ):
        super(RibbonIk, self).__init__(mod_name, desc, side)

        self.side = side
        self.jnt_amt = jnt_amt
        if parent:
            self.meta.set_parent(parent)

        self.still_grp = self.init_dag(
            mrc.Null(), self.side, ("still",), "grp"
        )
        if still_parent:
            self.still_grp.set_parent(still_parent)

        self.len = om.MVector(
            om.MVector(*end.world_pos) - om.MVector(*root.world_pos)
        ).length()

        # Main Controllers
        def _add_tip_ctrl(
            name: str,
        ) -> Tuple[mrc.Dag, mrc.Dag, mrc.Dag]:
            """Root/End controller"""
            ctrl = self.init_dag(
                mrc.Controller(mrc.Cp.plus), self.side, (name,), "ctrl"
            )
            ctrl.attr("ro").v = mrc.Ro.yzx
            ctrl.lhattrs("r", "s", "v")
            ctrl.color = ctrl_clr
            ctrlzr = self.init_dag(
                mrc.group(ctrl), self.side, (name, "ctrl"), "zr"
            )
            ctrlaim = self.init_dag(
                mrc.Null(), self.side, (name, "ctrl"), "aim"
            )

            ctrlaim.set_parent(ctrl)
            ctrlzr.set_parent(self.meta)

            return ctrl, ctrlzr, ctrlaim

        (
            self.root_ctrl,
            self.root_ctrl_zr,
            self.root_ctrl_aim,
        ) = _add_tip_ctrl("root")
        (
            self.end_ctrl,
            self.end_ctrl_zr,
            self.end_ctrl_aim,
        ) = _add_tip_ctrl("end")
        self.end_ctrl_zr.attr("ty").v = self.len

        self.mid_ctrl = self.init_dag(
            mrc.Controller(ctrl_shp), self.side, ("mid",), "ctrl"
        )
        self.mid_ctrl.attr("ro").v = mrc.Ro.yzx
        self.mid_ctrl.lhattrs("s", "v")
        self.mid_ctrl.color = ctrl_clr

        self.root_ctrl.scale_shape(self.len / 10.00)
        self.end_ctrl.scale_shape(self.len / 10.00)
        self.mid_ctrl.scale_shape(self.len / 5.00)

        self.mid_ctrl_ofst = self.init_dag(
            mrc.group(self.mid_ctrl), self.side, ("mid", "ctrl"), "ofst"
        )
        self.mid_ctrl_aim = self.init_dag(
            mrc.group(self.mid_ctrl_ofst), self.side, ("mid", "ctrl"), "aim"
        )
        self.mid_ctrl_zr = self.init_dag(
            mrc.group(self.mid_ctrl_aim), self.side, ("mid", "ctrl"), "zr"
        )
        self.mid_ctrl_zr.set_parent(self.meta)
        mrc.point_constraint(self.root_ctrl, self.end_ctrl, self.mid_ctrl_zr)

        self.auto_squash = self.mid_ctrl.add(
            ln="autoSquash", min=0, max=1, k=True
        )
        self.squash = self.mid_ctrl.add(ln="squash", k=True)

        # Aims
        mrc.aim_constraint(
            self.end_ctrl,
            self.root_ctrl_aim,
            aim=(0, 1, 0),
            u=(1, 0, 0),
            wut="objectrotation",
            wuo=self.meta,
            wu=(1, 0, 0),
        )
        mrc.aim_constraint(
            self.root_ctrl,
            self.end_ctrl_aim,
            aim=(0, -1, 0),
            u=(1, 0, 0),
            wut="objectrotation",
            wuo=self.meta,
            wu=(1, 0, 0),
        )
        mrc.aim_constraint(
            self.end_ctrl,
            self.mid_ctrl_aim,
            aim=(0, 1, 0),
            u=(1, 0, 0),
            wut="objectrotation",
            wuo=self.meta,
            wu=(1, 0, 0),
        )

        # Vectors
        def _create_vector_node(
            name: str, hierarchy: Tuple[mrc.Dag]
        ) -> mrc.Attr:
            pma = self.init_node(
                mrc.create("plusMinusAverage"),
                self.side,
                (name, "vec"),
                "pma",
            )
            for ix, dag in enumerate(hierarchy):
                dag.attr("t") >> pma.attr("i3[{}]".format(ix))

            return pma.attr("o3")

        self.root_vec = _create_vector_node(
            "root", (self.root_ctrl_zr, self.root_ctrl)
        )
        self.end_vec = _create_vector_node(
            "end", (self.end_ctrl_zr, self.end_ctrl)
        )
        self.mid_vec = _create_vector_node(
            "mid",
            (
                self.mid_ctrl_zr,
                self.mid_ctrl_aim,
                self.mid_ctrl_ofst,
                self.mid_ctrl,
            ),
        )

        # Lengthes
        def _add_len(name: str, from_: mrc.Attr, to_: mrc.Attr) -> mrc.Attr:
            dist = self.init_node(
                mrc.create("distanceBetween"), self.side, (name,), "dist"
            )
            from_ >> dist.attr("p1")
            to_ >> dist.attr("p2")

            return dist.attr("d")

        self.line_len = _add_len("line", self.root_vec, self.end_vec)
        self.bottom_len = _add_len("bottom", self.root_vec, self.mid_vec)
        self.top_len = _add_len("top", self.mid_vec, self.end_vec)
        self.orig_len = self.line_len.v

        # Curves
        def _add_curve(names: Tuple[str], offset: float) -> mrc.Dag:
            endpos = end.world_pos
            endpos[2] = endpos[2] + offset
            crvstr = mc.curve(d=1, p=((0, 0, offset), endpos))
            crvstr = mc.rebuildCurve(crvstr, ch=False, d=3, s=2, kr=0)[0]

            crv = self.init_dag(mrc.Dag(crvstr), self.side, names, "crv")
            crv.set_parent(self.still_grp)

            return crv

        self.crv = _add_curve(("",), 0)
        self.up_crv = _add_curve(("up",), self.len * -0.125)
        self.still_grp.hide = True

        # Clusters
        def _create_cluster(
            name: str, affected_compo: str
        ) -> Tuple[mrc.Dag, mrc.Dag, mrc.Dag]:
            clus_name, clushand_name = mc.cluster(affected_compo)
            clus = self.init_node(
                mrc.Node(clus_name), self.side, (name,), "cluster"
            )
            clus_hand = self.init_dag(
                mrc.Dag(clushand_name), self.side, (name,), "clusterHandle"
            )
            clus_hand_zr = self.init_dag(
                mrc.group(clus_hand), self.side, (name, "clus", "hand"), "zr"
            )
            clus_hand.hide = True

            return clus, clus_hand, clus_hand_zr

        # Bend Cluster
        (
            self.bend_clus,
            self.bend_clus_hand,
            self.bend_clus_hand_zr,
        ) = _create_cluster("bend", (self.crv.name, self.up_crv.name))
        self.bend_clus_hand_zr.set_parent(self.still_grp)

        self.mid_ctrl.attr("tx") >> self.bend_clus_hand.attr("tx")
        self.mid_ctrl.attr("tz") >> self.bend_clus_hand.attr("tz")
        self.mid_ctrl.attr("r") >> self.bend_clus_hand.attr("r")

        self.bend_clus_ty_div = self.init_node(
            mrc.create("multiplyDivide"),
            self.side,
            ("bend", "clus", "ty"),
            "div",
        )
        self.bend_clus_ty_div.attr("op").v = mrc.Op.divide
        self.bend_clus_ty_div.attr("i1x").v = self.orig_len
        self.line_len >> self.bend_clus_ty_div.attr("i2x")

        self.bend_clus_ty_mult = self.init_node(
            mrc.create("multDoubleLinear"),
            self.side,
            ("bend", "clus", "ty"),
            "mult",
        )
        self.bend_clus_ty_div.attr("ox") >> self.bend_clus_ty_mult.attr("i1")
        self.mid_ctrl.attr("ty") >> self.bend_clus_ty_mult.attr("i2")
        self.bend_clus_ty_mult.attr("o") >> self.bend_clus_hand.attr("ty")

        self.bend_clus.attr("wl[0].w[0]").v = 0
        self.bend_clus.attr("wl[1].w[0]").v = 0
        self.bend_clus.attr("wl[0].w[1]").v = 0.4
        self.bend_clus.attr("wl[1].w[1]").v = 0.4
        self.bend_clus.attr("wl[0].w[3]").v = 0.4
        self.bend_clus.attr("wl[1].w[3]").v = 0.4
        self.bend_clus.attr("wl[0].w[4]").v = 0
        self.bend_clus.attr("wl[1].w[4]").v = 0

        # Xform Cluster
        (
            self.xform_clus,
            self.xform_clus_hand,
            self.xform_clus_hand_zr,
        ) = _create_cluster("xform", (self.crv.name, self.up_crv.name))
        self.xform_clus_hand_zr.set_parent(self.still_grp)

        self.xform_clus_sc_div = self.init_node(
            mrc.create("multiplyDivide"),
            self.side,
            ("xform", "clus", "sc"),
            "div",
        )
        self.xform_clus_sc_div.attr("op").v = mrc.Op.divide
        self.line_len >> self.xform_clus_sc_div.attr("i1x")
        self.xform_clus_sc_div.attr("i2x").v = self.orig_len
        self.xform_clus_sc_div.attr("ox") >> self.xform_clus_hand.attr("sy")

        # Orient rig
        self.meta.snap(root)
        mru.aim(self.meta, end, (0, 1, 0), (1, 0, 0), root.x_axis)

        # Ribbon
        self.ribbon = RibbonCurve(
            crv=self.crv,
            up_crv=self.up_crv,
            jnt_amt=self.jnt_amt,
            mod_name="{}_rib".format(mod_name),
            desc=desc,
            side=self.side,
            ctrl_shp=dtl_ctrl_shp,
            ctrl_clr=dtl_clr,
            parent=self.mid_ctrl_aim,
            skin_parent=skin_parent,
        )
        self.mid_ctrl.attr("autoSquash") >> self.ribbon.meta.attr("autoSquash")
        self.mid_ctrl.attr("squash") >> self.ribbon.meta.attr("squash")
