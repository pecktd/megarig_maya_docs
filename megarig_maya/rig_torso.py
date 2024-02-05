from typing import Union, Tuple

import maya.cmds as mc
import maya.OpenMaya as om

from megarig_maya import core as mrc
from megarig_maya import rig_base
from megarig_maya import utils as mru
from megarig_maya import rig_main_group
from megarig_maya import rig_motion_path_spline_ik


class TorsoRig(rig_base.BaseRig):
    def __init__(
        self,
        mod_desc: str,
        hip_guide: str,
        pelvis_guide: str,
        mid_spine_guide: str,
        chest_guide: str,
        parent: mrc.Dag,
        skin_parent: mrc.Dag,
        main_grp: rig_main_group.MainGroup,
    ):
        super(TorsoRig, self).__init__("torso", mod_desc, mrc.Side.none)
        if parent:
            self.meta.snap(parent)
            self.meta.set_parent(parent)

        self.hip_guide = mrc.Dag(hip_guide)
        self.pelvis_guide = mrc.Dag(pelvis_guide)
        self.mid_spine_guide = mrc.Dag(mid_spine_guide)
        self.chest_guide = mrc.Dag(chest_guide)

        self.pelvis_locpos = mru.mvec_to_tuple(
            self.pelvis_guide.world_vec - skin_parent.world_vec
        )
        self.mid_spine_locpos = mru.mvec_to_tuple(
            self.mid_spine_guide.world_vec - skin_parent.world_vec
        )
        self.chest_locpos = mru.mvec_to_tuple(
            self.chest_guide.world_vec - skin_parent.world_vec
        )

        self.still_grp = self.init_dag(
            mrc.Null(), self.side, ("still",), "grp"
        )
        self.still_grp.hide = True
        if main_grp:
            self.still_grp.set_parent(main_grp.still_grp)

        self.spine_len = om.MVector(
            om.MVector(*self.chest_guide.world_pos)
            - om.MVector(*self.pelvis_guide.world_pos)
        ).length()

        # Main Controllers
        def _add_ctrl(
            name: str,
            guide: mrc.Dag,
            shape: mrc.Cp,
            parent: mrc.Dag,
            col: mrc.Color,
            add_cons: bool,
        ) -> Tuple[mrc.Dag, mrc.Dag, mrc.Dag, mrc.Dag]:
            (
                ctrl,
                ofst,
                ori,
                zr,
            ) = self.add_controller(
                mrc.Side.none,
                (name,),
                shape,
                col,
                ["s", "v"],
            )
            ctrl.rotate_order = mrc.Ro.xzy
            zr.snap(guide)
            zr.set_parent(parent)

            rigjnt = self.init_dag(
                mru.create_joint_at(ctrl), self.side, (name,), "rigJnt"
            )
            rigjnt.set_parent(self.meta)
            rigjnt.attr("ds").v = mrc.Ds.none

            locjnt = self.init_dag(
                mru.create_joint_at(rigjnt), self.side, (name,), "locJnt"
            )
            locjnt.set_parent(self.still_grp)

            if add_cons:
                cons = mrc.parent_constraint(ctrl, rigjnt)
                cons.set_parent(self.still_grp)

            rigjnt.attr("t") >> locjnt.attr("t")
            rigjnt.attr("r") >> locjnt.attr("r")

            return ctrl, ofst, ori, zr, rigjnt, locjnt

        (
            self.pelvis_ctrl,
            self.pelvis_ctrl_ofst,
            self.pelvis_ctrl_ori,
            self.pelvis_ctrl_zr,
            self.pelvis_rj,
            self.pelvis_lj,
        ) = _add_ctrl(
            "pelvis",
            self.pelvis_guide,
            mrc.Cp.cube,
            self.meta,
            mrc.Color.yellow,
            False,
        )
        pelv_pntcons = mrc.pntcons(self.pelvis_ctrl, self.pelvis_rj)
        pelv_pntcons.set_parent(self.still_grp)

        pelv_spfoll = self.pelvis_ctrl.add(
            ln="spineFollow", min=0, max=1, dv=0.5, k=True
        )
        pelv_oricons = mrc.oricons(
            self.pelvis_ctrl_ofst, self.pelvis_ctrl, self.pelvis_rj
        )
        pelv_spfoll_rev = self.init_node(
            mrc.create("reverse"), self.side, ("sp", "foll"), "rev"
        )
        pelv_spfoll >> pelv_oricons.attr("w1")
        pelv_spfoll >> pelv_spfoll_rev.attr("ix")
        pelv_spfoll_rev.attr("ox") >> pelv_oricons.attr("w0")

        self.hip_grp = self.init_dag(mrc.Null(), self.side, ("hip",), "grp")
        self.hip_grp.snap(self.hip_guide)
        self.hip_grp.set_parent(self.pelvis_ctrl)

        (
            self.spine_1_ctrl,
            self.spine_1_ctrl_ofst,
            self.spine_1_ctrl_ori,
            self.spine_1_ctrl_zr,
            self.spine_1_rj,
            self.spine_1_lj,
        ) = _add_ctrl(
            "spine_1",
            self.pelvis_guide,
            mrc.Cp.nurbs_circle,
            self.meta,
            mrc.Color.soft_yellow,
            True,
        )
        self.spine_1_ctrl.lhattrs("t")

        (
            self.spine_2_ctrl,
            self.spine_2_ctrl_ofst,
            self.spine_2_ctrl_ori,
            self.spine_2_ctrl_zr,
            self.spine_2_rj,
            self.spine_2_lj,
        ) = _add_ctrl(
            "spine_2",
            self.mid_spine_guide,
            mrc.Cp.nurbs_circle,
            self.spine_1_ctrl,
            mrc.Color.soft_yellow,
            True,
        )
        self.spine_2_ctrl.lhattrs("t")

        (
            self.chest_ctrl,
            self.chest_ctrl_ofst,
            self.chest_ctrl_ori,
            self.chest_ctrl_zr,
            self.chest_rj,
            self.chest_lj,
        ) = _add_ctrl(
            "chest",
            self.chest_guide,
            mrc.Cp.cube,
            self.meta,
            mrc.Color.yellow,
            True,
        )
        mrc.parcons(self.spine_2_ctrl, self.chest_ctrl_ori, mo=True)

        (
            self.mid_ctrl,
            self.mid_ctrl_ofst,
            self.mid_ctrl_ori,
            self.mid_ctrl_zr,
            self.mid_rj,
            self.mid_lj,
        ) = _add_ctrl(
            "mid",
            self.meta,
            mrc.Cp.cube,
            self.meta,
            mrc.Color.dark_yellow,
            True,
        )

        # Curves
        def _add_curve(names: Tuple[str], offset: float) -> mrc.Dag:
            poses = [
                list(self.pelvis_locpos),
                list(self.mid_spine_locpos),
                list(self.chest_locpos),
            ]

            for p in poses:
                p[0] = p[0] + offset

            crvstr = mc.curve(d=1, p=poses)
            crvstr = mc.rebuildCurve(crvstr, ch=False, d=3, s=2, kr=0)[0]

            crv = self.init_dag(mrc.Dag(crvstr), self.side, names, "crv")
            crv.set_parent(self.still_grp)

            return crv

        self.mid_crv = _add_curve(("mid",), 0)
        self.mid_up_crv = _add_curve(("mid_up",), 0.033 * self.spine_len)

        # Mid Curves
        def _bind_mid_curve(name: str, crv: mrc.Dag) -> mrc.Node:
            skin = self.init_node(
                mrc.Node(
                    mc.skinCluster(self.pelvis_lj, self.chest_lj, crv)[0]
                ),
                self.side,
                (name,),
                "skc",
            )
            skin.attr("wl[0].w[0]").v = 1
            skin.attr("wl[0].w[1]").v = 0
            skin.attr("wl[1].w[0]").v = 1
            skin.attr("wl[1].w[01]").v = 0
            skin.attr("wl[2].w[0]").v = 0.5
            skin.attr("wl[2].w[1]").v = 0.5
            skin.attr("wl[3].w[0]").v = 0
            skin.attr("wl[3].w[1]").v = 1
            skin.attr("wl[4].w[0]").v = 0
            skin.attr("wl[4].w[1]").v = 1

            return skin

        self.mid_crv_skc = _bind_mid_curve("mid", self.mid_crv)
        self.mid_up_crv_skc = _bind_mid_curve("mid_up", self.mid_up_crv)

        def _add_mid_mp(name: str, shape: mrc.Dag) -> mrc.Node:
            mp = self.init_node(
                mrc.create("motionPath"), self.side, ("mid", name), "mp"
            )
            shape.attr("ws[0]") >> mp.attr("gp")
            mp.attr("fm").v = True
            mp.attr("u").v = 0.5

            return mp

        self.mid_mp = _add_mid_mp("", self.mid_crv.get_shape())
        self.mid_up_mp = _add_mid_mp("up", self.mid_up_crv.get_shape())

        mid_up_sub, mid_up_norm = mru.create_vector_node(
            self.mid_up_mp.attr("ac"), self.mid_mp.attr("ac")
        )
        self.mid_up_sub = self.init_node(
            mid_up_sub, self.side, ("mid", "up"), "sub"
        )
        self.mid_up_norm = self.init_node(
            mid_up_norm, self.side, ("mid", "up"), "norm"
        )
        self.mid_mp.attr("fa").v = 1
        self.mid_mp.attr("ua").v = 0
        self.mid_mp.attr("iu").v = True
        self.mid_up_norm.attr("o") >> self.mid_mp.attr("wu")
        self.mid_mp.attr("ac") >> self.mid_ctrl_zr.attr("t")
        self.mid_mp.attr("r") >> self.mid_ctrl_zr.attr("r")

        # IK Curves
        self.crv = _add_curve(("",), 0)
        self.up_crv = _add_curve(("up",), 0.033 * self.spine_len)

        def _bind_curve(name: str, crv: mrc.Dag) -> mrc.Node:
            skin = self.init_node(
                mrc.Node(
                    mc.skinCluster(
                        self.pelvis_lj, self.mid_lj, self.chest_lj, crv
                    )[0]
                ),
                self.side,
                (name,),
                "skc",
            )
            skin.attr("wl[0].w[0]").v = 1
            skin.attr("wl[0].w[1]").v = 0
            skin.attr("wl[0].w[2]").v = 0
            skin.attr("wl[1].w[0]").v = 0.7
            skin.attr("wl[1].w[1]").v = 0.3
            skin.attr("wl[1].w[2]").v = 0
            skin.attr("wl[2].w[0]").v = 0
            skin.attr("wl[2].w[1]").v = 1
            skin.attr("wl[2].w[2]").v = 0
            skin.attr("wl[3].w[0]").v = 0
            skin.attr("wl[3].w[1]").v = 0.3
            skin.attr("wl[3].w[2]").v = 0.7
            skin.attr("wl[4].w[0]").v = 0
            skin.attr("wl[4].w[1]").v = 0
            skin.attr("wl[4].w[2]").v = 1

            return skin

        self.crv_skc = _bind_curve("", self.crv)
        self.up_crv_skc = _bind_curve("up", self.up_crv)

        # Auto-Lengthes
        mru.add_divide_attr(self.chest_ctrl, "autoLengthes")
        self.mn_len_attr = self.chest_ctrl.add(
            ln="mnLen", dv=0.9, min=0.1, max=1, k=True
        )
        self.mx_len_attr = self.chest_ctrl.add(
            ln="mxLen", dv=1.1, min=1, k=True
        )

        self.mx_len_mult = self.init_node(
            mrc.create("multDoubleLinear"), self.side, ("mx", "len"), "mult"
        )
        self.mx_len_mult.attr("i1").v = self.spine_len
        self.mx_len_attr >> self.mx_len_mult.attr("i2")

        self.mn_len_mult = self.init_node(
            mrc.create("multDoubleLinear"), self.side, ("mn", "len"), "mult"
        )
        self.mn_len_mult.attr("i1").v = self.spine_len
        self.mn_len_attr >> self.mn_len_mult.attr("i2")

        def _add_auto_length(
            name: str,
            ctrl: mrc.Dag,
            ctrl_ori: mrc.Dag,
            ctrl_zr: mrc.Dag,
            tar_zr: mrc.Dag,
            tar_ofst: mrc.Dag,
        ) -> None:
            curr_pos = self.init_node(
                mrc.create("plusMinusAverage"), self.side, (name, "pos"), "sum"
            )
            ctrl.attr("t") >> curr_pos.attr("i3[0]")
            ctrl_zr.attr("t") >> curr_pos.attr("i3[1]")

            curr_vec, curr_norm_vec = mru.create_vector_node(
                tar_zr.attr("t"), curr_pos.attr("o3")
            )
            curr_vec = self.init_node(
                curr_vec, self.side, (name, "vec"), "sub"
            )
            curr_norm_vec = self.init_node(
                curr_norm_vec, self.side, (name, "vec"), "norm"
            )

            curr_dist = self.init_node(
                mrc.create("distanceBetween"), self.side, (name,), "dist"
            )
            curr_vec.attr("o3") >> curr_dist.attr("p2")

            def _add_mv_cond(
                mnmx: str, len_mult: mrc.Node, op: mrc.Op
            ) -> mrc.Node:
                mvlen_mult = self.init_node(
                    mrc.create("plusMinusAverage"),
                    self.side,
                    (name, mnmx, "mv", "mult"),
                    "sum",
                )
                mvlen_mult.attr("op").v = mrc.Op.subtract
                curr_dist.attr("d") >> mvlen_mult.attr("i1[0]")
                len_mult.attr("o") >> mvlen_mult.attr("i1[1]")

                mvlen_val = self.init_node(
                    mrc.create("multiplyDivide"),
                    self.side,
                    (name, mnmx, "mv", "val"),
                    "mult",
                )
                mvlen_mult.attr("o1") >> mvlen_val.attr("i1x")
                mvlen_mult.attr("o1") >> mvlen_val.attr("i1y")
                mvlen_mult.attr("o1") >> mvlen_val.attr("i1z")
                curr_norm_vec.attr("o") >> mvlen_val.attr("i2")

                cond = self.init_node(
                    mrc.create("condition"),
                    self.side,
                    (name, mnmx, "mv"),
                    "cond",
                )
                cond.attr("op").v = op
                curr_dist.attr("d") >> cond.attr("ft")
                len_mult.attr("o") >> cond.attr("st")
                cond.attr("cfr").v = 0
                mvlen_val.attr("o") >> cond.attr("ct")
                cond.attr("cf").v = (0, 0, 0)

                return cond

            mx_cond = _add_mv_cond("mx", self.mx_len_mult, mrc.Op.greater_than)
            mn_cond = _add_mv_cond("mn", self.mn_len_mult, mrc.Op.less_than)

            mv_sum = self.init_node(
                mrc.create("plusMinusAverage"), self.side, (name, "mv"), "sum"
            )
            mx_cond.attr("oc") >> mv_sum.attr("i3[0]")
            mn_cond.attr("oc") >> mv_sum.attr("i3[1]")
            mv_sum.attr("o3") >> tar_ofst.attr("t")

        _add_auto_length(
            "chest",
            self.chest_ctrl,
            self.chest_ctrl_ori,
            self.chest_ctrl_zr,
            self.pelvis_ctrl_zr,
            self.pelvis_ctrl_ofst,
        )
        _add_auto_length(
            "pelv",
            self.pelvis_ctrl,
            self.pelvis_ctrl_ori,
            self.pelvis_ctrl_zr,
            self.chest_ctrl_zr,
            self.chest_ctrl_ofst,
        )

        # Spline IK
        self.spline = rig_motion_path_spline_ik.SplineIk(
            crv=self.crv,
            up_crv=self.up_crv,
            aim_axis="y",
            up_axis="x",
            jnt_amt=6,
            mod_name="torso_sp_ik",
            desc="",
            side=mrc.Side.none,
            jnt_type="ikJnt",
        )
        self.spline.meta.snap(self.meta)
        self.spline.meta.set_parent(self.meta)
        self.spline.master_jnt.hide = True
        self.spline.still_grp.set_parent(self.still_grp)

        # Skin Jnts
        def _add_skin_jnt(
            name: str, target: mrc.Dag, parent: mrc.Dag, add_cons: bool
        ) -> mrc.Dag:
            jnt = self.init_dag(
                mru.create_joint_at(target), self.side, (name,), "jnt"
            )
            if add_cons:
                cons = mrc.parent_constraint(target, jnt)
                cons.set_parent(self.still_grp)
            jnt.set_parent(parent)
            self.add_skin_joints_to((jnt,), mrc.SKIN_SET, main_grp.jnt_vis)

            return jnt

        self.hip_jnt = _add_skin_jnt("hip", self.hip_grp, skin_parent, True)
        self.pelvis_jnt = _add_skin_jnt(
            "pelvis", self.pelvis_rj, self.hip_jnt, True
        )
        self.spine_1_jnt = _add_skin_jnt(
            "spine_1", self.spline.jnts[0], self.pelvis_jnt, True
        )
        self.spine_2_jnt = _add_skin_jnt(
            "spine_2", self.spline.jnts[1], self.spine_1_jnt, True
        )
        self.spine_3_jnt = _add_skin_jnt(
            "spine_3", self.spline.jnts[2], self.spine_2_jnt, True
        )
        self.spine_4_jnt = _add_skin_jnt(
            "spine_4", self.spline.jnts[3], self.spine_3_jnt, True
        )
        self.spine_5_jnt = _add_skin_jnt(
            "spine_5", self.spline.jnts[4], self.spine_4_jnt, True
        )
        self.spine_6_jnt = _add_skin_jnt(
            "spine_6", self.spline.jnts[5], self.spine_5_jnt, True
        )
        self.chest_jnt = _add_skin_jnt(
            "chest", self.chest_ctrl, self.spine_6_jnt, False
        )

        chest_oricons = mrc.oricons(self.chest_ctrl, self.chest_jnt)
        chest_oricons.set_parent(self.still_grp)
