from typing import Union, Tuple

from megarig_maya import core as mrc
from megarig_maya import utils as mru
from megarig_maya import naming_tools as mrnt
from megarig_maya import rig_base


class SplineIk(rig_base.BaseRig):
    """Spline IK with motion path nodes"""

    def __init__(
        self,
        crv: mrc.Dag,
        up_crv: mrc.Dag,
        aim_axis: str,
        up_axis: str,
        jnt_amt: int,
        mod_name: str,
        desc: str,
        side: mrc.Side,
        jnt_type: str,
    ):
        # Main Groups
        super(SplineIk, self).__init__(mod_name, desc, side)

        self.crv = crv
        self.up_crv = up_crv

        self.crv_shp = self.crv.get_shape()
        self.up_crv_shp = self.up_crv.get_shape()

        if not jnt_type:
            jnt_type = "jnt"

        self.still_grp = self.init_dag(
            mrc.Null(), self.side, ("still",), "grp"
        )

        index_param = [(i, i * (1 / (jnt_amt - 1))) for i in range(jnt_amt)]

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

        # Stretch Attrs
        mru.add_divide_attr(self.meta, "stretch")
        self.mx_str = self.meta.add(ln="mxStr", min=1, dv=2, k=True)

        self.mx_abs_str = self.init_node(
            mrc.create("addDoubleLinear"),
            self.side,
            ("mx", "abs", "str"),
            "add",
        )
        self.mx_str >> self.mx_abs_str.attr("i1")
        self.mx_abs_str.attr("i2").v = -1

        # Stretch - Length Ratio
        self.len_rat_div = self.init_node(
            mrc.create("multiplyDivide"), self.side, ("len", "rat"), "div"
        )
        self.len_rat_div.attr("op").v = mrc.Op.divide
        self.curr_len >> self.len_rat_div.attr("i1x")
        self.orig_len >> self.len_rat_div.attr("i2x")

        self.len_rat = self.len_rat_div.attr("ox")

        # Point On Curve
        self.pocs = []
        self.mps = []
        self.up_mps = []

        def _add_poc(index: int, param: float) -> None:
            idstr = str(index + 1)
            poc = self.init_dag(mrc.Null(), self.side, ("", idstr), "poc")
            poc.set_parent(self.meta)

            def _add_mp(shp: mrc.Dag, names: Tuple[str]) -> mrc.Node:
                mp = self.init_node(
                    mrc.create("motionPath"), self.side, names, "mp"
                )
                shp.attr("ws[0]") >> mp.attr("gp")
                mp.attr("fm").v = True

                return mp

            mp = _add_mp(self.crv_shp, (idstr,))
            ump = _add_mp(self.up_crv_shp, ("up", idstr))

            # Curve Parameter
            curr_param = self.init_node(
                mrc.create("multiplyDivide"),
                self.side,
                ("curr", "param", idstr),
                "div",
            )
            curr_param.attr("op").v = mrc.Op.divide
            curr_param.attr("i1x").v = param
            self.len_rat >> curr_param.attr("i2x")

            mx_str_param = self.init_node(
                mrc.create("multDoubleLinear"),
                self.side,
                ("mx", "str", idstr),
                "mult",
            )
            self.mx_str >> mx_str_param.attr("i1")
            curr_param.attr("ox") >> mx_str_param.attr("i2")

            str_clamp = self.init_node(
                mrc.create("clamp"), self.side, ("str", idstr), "clamp"
            )
            str_clamp.attr("ipr").v = param
            curr_param.attr("ox") >> str_clamp.attr("mnr")
            mx_str_param.attr("o") >> str_clamp.attr("mxr")

            str_cond = self.init_node(
                mrc.create("condition"), self.side, ("str", idstr), "cond"
            )
            str_cond.attr("op").v = mrc.Operator.greater_than
            self.len_rat >> str_cond.attr("ft")
            str_cond.attr("st").v = 1
            str_clamp.attr("opr") >> str_cond.attr("ctr")
            str_cond.attr("cfr").v = (1 / (jnt_amt - 1)) * index
            str_cond.attr("ocr") >> mp.attr("u")
            str_cond.attr("ocr") >> ump.attr("u")

            self.mps.append(mp)
            self.up_mps.append(ump)
            self.pocs.append(poc)

        for i, p in index_param:
            _add_poc(index=i, param=p)

        # Joints
        self.master_jnt = self.init_dag(
            mrc.Joint(), self.side, ("master",), jnt_type
        )
        self.master_jnt.attr("ds").v = mrc.DrawStyle.none
        self.master_jnt.set_parent(self.meta)
        self.jnts = []

        def _add_joint(
            index: int,
            param: float,
            poc: mrc.Dag,
            mp: mrc.Node,
            ump: mrc.Node,
            nmp: Union[mrc.Node, None],
        ) -> None:
            idstr = str(index + 1)

            def _create_vector_node(
                name: str, in1: mrc.Attr, in2: mrc.Attr
            ) -> mrc.Node:
                """Creats a vector from in1 to in2"""
                sub = self.init_node(
                    mrc.create("plusMinusAverage"),
                    self.side,
                    (name, "vec", idstr),
                    "sub",
                )
                sub.attr("op").v = mrc.Op.subtract

                norm = self.init_node(
                    mrc.create("vectorProduct"),
                    self.side,
                    (name, "vec", idstr),
                    "norm",
                )
                norm.attr("op").v = mrc.Op.no
                norm.attr("no").v = True

                in2 >> sub.attr("i3[0]")
                in1 >> sub.attr("i3[1]")
                sub.attr("o3") >> norm.attr("i1")

                return norm

            def _tip_jnt(mp: mrc.Node, ump: mrc.Node):
                axint = {"x": 0, "y": 1, "z": 2}
                vvec = _create_vector_node("v", mp.attr("ac"), ump.attr("ac"))

                mp.attr("fa").v = axint[aim_axis]
                mp.attr("ua").v = axint[up_axis]

                vvec.attr("o") >> mp.attr("wu")

            def _none_tip_jnt(
                mp: mrc.Node, nmp: mrc.Node, ump: mrc.Node
            ) -> mrc.Attr:
                """Uses vectors from motionPath nodes to defien the rotation"""
                uvec = _create_vector_node("u", mp.attr("ac"), nmp.attr("ac"))
                upvec = _create_vector_node(
                    "up", mp.attr("ac"), ump.attr("ac")
                )
                wvec = self.init_node(
                    mrc.create("vectorProduct"),
                    self.side,
                    ("w", "vec", idstr),
                    "cross",
                )
                wvec.attr("op").v = mrc.Op.cross
                upvec.attr("o") >> wvec.attr("i1")
                uvec.attr("o") >> wvec.attr("i2")

                vvec = self.init_node(
                    mrc.create("vectorProduct"),
                    self.side,
                    ("v", "vec", idstr),
                    "cross",
                )
                vvec.attr("op").v = mrc.Op.cross
                uvec.attr("o") >> vvec.attr("i1")
                wvec.attr("o") >> vvec.attr("i2")

                fbf = self.init_node(
                    mrc.create("fourByFourMatrix"),
                    self.side,
                    ("xform", idstr),
                    "fbf",
                )
                axmat = {
                    "x": ("i00", "i01", "i02"),
                    "y": ("i10", "i11", "i12"),
                    "z": ("i20", "i21", "i22"),
                }
                wax = "xyz".replace(aim_axis, "").replace(up_axis, "")

                uvec.attr("ox") >> fbf.attr(axmat[aim_axis][0])
                uvec.attr("oy") >> fbf.attr(axmat[aim_axis][1])
                uvec.attr("oz") >> fbf.attr(axmat[aim_axis][2])
                vvec.attr("ox") >> fbf.attr(axmat[up_axis][0])
                vvec.attr("oy") >> fbf.attr(axmat[up_axis][1])
                vvec.attr("oz") >> fbf.attr(axmat[up_axis][2])
                wvec.attr("ox") >> fbf.attr(axmat[wax][0])
                wvec.attr("oy") >> fbf.attr(axmat[wax][1])
                wvec.attr("oz") >> fbf.attr(axmat[wax][2])
                mp.attr("xc") >> fbf.attr("i30")
                mp.attr("yc") >> fbf.attr("i31")
                mp.attr("zc") >> fbf.attr("i32")

                decomp_mat = self.init_node(
                    mrc.create("decomposeMatrix"),
                    self.side,
                    ("xform", idstr),
                    "decompMat",
                )
                poc.attr("ro") >> decomp_mat.attr("ro")
                fbf.attr("o") >> decomp_mat.attr("imat")

                decomp_mat.attr("ot") >> poc.attr("t")
                decomp_mat.attr("or") >> poc.attr("r")

                return decomp_mat.attr("or")

            out_rot = mp.attr("r")
            if index != (jnt_amt - 1):
                out_rot = _none_tip_jnt(mp, nmp, ump)
            else:
                _tip_jnt(mp, ump)
                mp.attr("ac") >> poc.attr("t")

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
            tw_q2e = self.init_node(
                mrc.create("quatToEuler"), self.side, ("tw", idstr), "q2e"
            )

            out_rot >> mp_e2q.attr("inputRotate")
            tw_add.attr("o") >> tw_e2q.attr("iry")
            tw_e2q.attr("oq") >> quat_prod.attr("iq1")
            mp_e2q.attr("oq") >> quat_prod.attr("iq2")
            quat_prod.attr("oq") >> tw_q2e.attr("iq")
            poc.attr("ro") >> tw_q2e.attr("iro")
            tw_q2e.attr("outputRotate") >> poc.attr("r")

            # Joint
            jnt = self.init_dag(
                mru.create_joint_at(poc), self.side, (idstr,), jnt_type
            )
            jnt.freeze()
            if index:
                jnt.set_parent(self.jnts[index - 1])
            else:
                jnt.set_parent(self.master_jnt)
            self.jnts.append(jnt)

            cons = mrc.parent_constraint(poc, jnt)
            cons.set_parent(self.still_grp)

        for i, p in index_param:
            nmp = None
            if i < jnt_amt - 1:
                nmp = self.mps[i + 1]
            _add_joint(i, p, self.pocs[i], self.mps[i], self.up_mps[i], nmp)
