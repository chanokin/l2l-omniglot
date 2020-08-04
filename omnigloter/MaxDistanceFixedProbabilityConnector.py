from pynn_genn.connectors import DistanceDependentProbabilityConnector
from pynn_genn.random import NativeRNG
from pygenn import genn_model
import numpy as np
from scipy.stats import binom, norm
#  0     1
#  PROB, MAX_DIST,
#  2       3       4
#  PRE_NX, PRE_NY, PRE_NZ,
#  5       6       7
#  PRE_X0, PRE_Y0, PRE_Z0,
#  8       9       10
#  PRE_DX, PRE_DY, PRE_DZ,
#  11       12       13
#  POST_NX, POST_NY, POST_NZ,
#  14       15       16
#  POST_X0, POST_Y0, POST_Z0,
#  17       18       19
#  POST_DX, POST_DY, POST_DZ


class MaxDistanceFixedProbabilityConnector(DistanceDependentProbabilityConnector):
    __doc__ = DistanceDependentProbabilityConnector.__doc__

    def __init__(self, max_dist, probability, allow_self_connections=True,
                 rng=None, safe=True, callback=None):
        d_expr = "%s * ( d <= %s)" % (probability, max_dist)
        DistanceDependentProbabilityConnector.__init__(
            self, d_expr, allow_self_connections, rng, safe, callback)
        self.probability = probability
        self.max_dist = max_dist
        self._builtin_name = 'MaxDistanceFixedProbability'
        self.connectivity_init_possible = isinstance(rng, NativeRNG)
        self._needs_populations_shapes = True
        self.shapes = None

    @property
    def _conn_init_params(self):
        params = {
            'prob': self.probability,
            'max_dist': self.max_dist,
        }
        return dict(list(params.items()) + list(self.shapes.items()))

    def _init_connectivity(self):
        if self.connectivity_init_possible:
            return self._generate_init_snippet()

    def _generate_init_snippet(self):
        def _max_row_len(num_pre, num_post, pars):
            max_d = (2.0 * pars[1] + 1.0)
            max_conns = 1.0

            if pars[11] > 1:
                max_conns *= min(max_d / pars[17], pars[11])

            if pars[12] > 1:
                max_conns *= min(max_d / pars[18], pars[12])

            if pars[13] > 1:
                max_conns *= min(max_d / pars[19], pars[13])

            return int(binom.ppf(0.9999 ** (1.0 / num_pre),
                       n=max_conns, p=pars[0]))

        def _max_col_len(num_pre, num_post, pars):
            max_d = (2.0 * pars[1] + 1.0)
            max_conns = 1.0

            if pars[2] > 1:
                max_conns *= min(max_d / pars[8], pars[2])

            if pars[3] > 1:
                max_conns *= min(max_d / pars[9], pars[3])

            if pars[4] > 1:
                max_conns *= min(max_d / pars[10], pars[4])

            return int(binom.ppf(0.9999 ** (1.0 / num_post),
                       n=max_conns, p=pars[0]))

        _param_space = self._conn_init_params
        shp = self.shapes
        pre_per_row = int(shp['pre_nx'] * shp['pre_nz'])
        post_per_row = int(shp['post_nx'] * shp['post_nz'])
        delta_row = int(
            ((self.max_dist / shp['post_dy']) + 2) * post_per_row
        )
        n_post = shp['post_nx'] * shp['post_ny'] * shp['post_nz']
        names = [
            "prob", "max_dist",
            "pre_nx", "pre_ny", "pre_nz",
            "pre_x0", "pre_y0", "pre_z0",
            "pre_dx", "pre_dy", "pre_dz",
            "post_nx", "post_ny", "post_nz",
            "post_x0", "post_y0", "post_z0",
            "post_dx", "post_dy", "post_dz"
        ]
        state_vars = [
            # ("perRow", "int", per_row),
            # ("deltaRow", "int", delta_row),
            ("preRow", "int",
             "($(id_pre) / {}) * $(pre_dy) + $(pre_y0)".format(pre_per_row)),
            ("prevJ", "int",
             "max(-1,\n"
             "    (int)( ((preRow - $(post_y0)) / $(post_dy)) * {} - {} - 1 )\n"
             ")".format(post_per_row, delta_row)),
            ("endJ", "int",
             "min({}, \n"
             "    (int)( ((preRow - $(post_y0)) / $(post_dy)) * {} + {} + 1 )\n"
             ")".format(n_post, post_per_row, delta_row)),
        ]
        derived = [
            ("probLogRecip",
                genn_model.create_dpf_class(
                    lambda pars, dt: (1.0 / np.log(1.0 - pars[0])))()
             )
        ]

        _code = """
            #define toCoords(idx, nx, ny, nz, x, y, z) { \\
                int a = (int)(nx * nz);                  \\
                int inz = (int)nz;                       \\
                y = (float)(idx / a);                    \\
                x = (float)((idx - ((int)y * a)) / inz); \\
                z = (float)((idx - ((int)y * a)) % inz); \\
            }
            
            #define inDist(pre, post, output) { \\
                float pre_x, pre_y, pre_z, post_x, post_y, post_z; \\
                toCoords(pre, $(pre_nx), $(pre_ny), $(pre_nz), pre_x, pre_y, pre_z); \\
                toCoords(post, $(post_nx), $(post_ny), $(post_nz), post_x, post_y, post_z); \\
                pre_x = pre_x * $(pre_dx) + $(pre_x0); \\
                pre_y = pre_y * $(pre_dy) + $(pre_y0); \\
                pre_z = pre_z * $(pre_dz) + $(pre_z0); \\
                post_x = post_x * $(post_dx) + $(post_x0); \\
                post_y = post_y * $(post_dy) + $(post_y0); \\
                post_z = post_z * $(post_dz) + $(post_z0); \\
                float dx = post_x - pre_x, \\
                       dy = post_y - pre_y, \\
                       dz = post_z - pre_z; \\
                output = (sqrt((dx * dx) + (dy * dy) + (dz * dz)) <= ($(max_dist) * 1.4143)); \\
            }
                        
            const scalar u = $(gennrand_uniform);
            prevJ += (1 + (int)(log(u) * $(probLogRecip)));
            
            if(prevJ < endJ) {
                int out = 0;
                inDist($(id_pre), prevJ, out);
                if(out){
                    $(addSynapse, prevJ + $(id_post_begin));
                }
            }
            else {
                $(endRow);
            }

        """
        _snip = genn_model.create_custom_sparse_connect_init_snippet_class(
            "max_distance_fixed_probability",
            param_names=names,
            row_build_state_vars=state_vars,
            derived_params=derived,
            calc_max_row_len_func=genn_model.create_cmlf_class(_max_row_len)(),
            calc_max_col_len_func=genn_model.create_cmlf_class(_max_col_len)(),
            row_build_code=_code)

        return genn_model.init_connectivity(_snip, _param_space)


