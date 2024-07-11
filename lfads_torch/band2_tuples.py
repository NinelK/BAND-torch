from collections import namedtuple

SessionBatch = namedtuple(
    "SessionBatch",
    [
        "encod_data",
        "recon_data",
        "behavior",
        "ext_input",
        "truth",
        "sv_mask",
    ],
)

SessionOutput = namedtuple(
    "SessionOutput",
    [
        "output_params",
        "output_behavior_params",
        "noci_output_params",
        "noci_output_behavior_params",
        "factors",
        "noci_factors",
        "ic_mean",
        "ic_std",
        "co_means",
        "co_stds",
        "gen_states",
        "gen_init",
        "gen_inputs",
        "con_states",
    ],
)
