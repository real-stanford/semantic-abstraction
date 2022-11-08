import pandas as pd
import rich
import pickle
from dataset import synonyms
import numpy as np
from rich.console import Console
from rich.table import Table

test_objs = set(
    map(lambda l: l.rstrip().lstrip(), open("test_semantic_classes.txt", "r"))
)


def summarize_ovssc(metric="voxel32x32x32_iou"):
    ssc_approaches = {
        "Semantic Aware": pickle.load(
            open("models/semaware/ovssc/ovssc_eval_stats.pkl", "rb")
        ),
        "SemAbs + [Chefer et al]": pickle.load(
            open("models/chefer_et_al/ovssc/ovssc_eval_stats.pkl", "rb")
        ),
        "Ours": pickle.load(
            open(
                "models/ours/ovssc/ovssc_eval_stats.pkl",
                "rb",
            )
        ),
    }

    ovssc_stats = {
        "approach": [],
        "novel rooms": [],
        "novel visual": [],
        "novel vocab": [],
        "novel class": [],
    }
    pd.options.display.float_format = "{:,.3f}".format
    for approach, approach_stats in ssc_approaches.items():
        # approach_stats = approach_stats[approach_stats.label!='']
        approach_stats["room_id"] = approach_stats["scene_id"].apply(
            lambda s: int(s.split("_")[0].split("FloorPlan")[1])
        )
        approach_stats[metric] = approach_stats[metric] * 100
        cutoff_analysis = approach_stats.groupby("cutoff")[[metric]].mean()
        best_cutoff = cutoff_analysis[metric].idxmax()
        df = approach_stats[approach_stats.cutoff == best_cutoff]
        novel_class_mask = df.label.isin(test_objs)
        novel_vocab_mask = df.label.isin(synonyms.values())
        ovssc_stats["approach"].append(approach)

        novel_rooms_df = df[(df.split == "unseen_instances") & (~novel_class_mask)]
        mean_per_room = np.array(novel_rooms_df.groupby("room_id")[metric].mean())
        ovssc_stats["novel rooms"].append(mean_per_room.mean())

        novel_rooms_dr_df = df[
            (df.split == "unseen_instances_dr") & (~novel_class_mask)
        ]
        mean_per_room = np.array(novel_rooms_dr_df.groupby("room_id")[metric].mean())
        ovssc_stats["novel visual"].append(mean_per_room.mean())

        unseen_class_df = df[novel_class_mask]
        mean_per_label = unseen_class_df.groupby("label")[metric].mean()
        ovssc_stats["novel class"].append(np.array(mean_per_label).mean())

        unseen_vocab_df = df[
            (df.split == "unseen_instances_synonyms") & novel_vocab_mask
        ]
        mean_per_label = unseen_vocab_df.groupby("label")[metric].mean()
        ovssc_stats["novel vocab"].append(np.array(mean_per_label).mean())
    ovssc_stats = pd.DataFrame.from_dict(ovssc_stats)
    table = Table(title="OVSSC THOR", box=rich.box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Approach", justify="left")
    table.add_column("Novel Room", justify="right")
    table.add_column("Novel Visual", justify="right")
    table.add_column("Novel Vocab", justify="right")
    table.add_column("Novel Class", justify="right")
    for row in ovssc_stats.to_csv().split("\n")[1:-1]:
        approach, novel_room, novel_visual, novel_vocab, novel_class = row.split(",")[
            1:
        ]
        table.add_row(
            approach,
            f"{float(novel_room):.01f}",
            f"{float(novel_visual):.01f}",
            f"{float(novel_vocab):.01f}",
            f"{float(novel_class):.01f}",
            end_section=approach == "SemAbs + [Chefer et al]",
            style="green" if approach == "Ours" else "white",
        )
    console = Console()
    console.print(table)


def summarize_vool(metric="voxel32x32x32_iou"):
    vool_approaches = {
        "Semantic Aware": pickle.load(
            open("models/semaware/vool/vool_eval_stats.pkl", "rb")
        ),
        "ClipSpatial": pickle.load(
            open("models/clipspatial/vool/vool_eval_stats.pkl", "rb")
        ),
        "SemAbs + [Chefer et al]": pickle.load(
            open("models/chefer_et_al/vool/vool_eval_stats.pkl", "rb")
        ),
        "Ours": pickle.load(open("models/ours/vool/vool_eval_stats.pkl", "rb")),
    }
    vool_stats = {
        "approach": [],
        "relation": [],
        "novel rooms": [],
        "novel visual": [],
        "novel vocab": [],
        "novel class": [],
    }
    relations = vool_approaches["Ours"].spatial_relation_name.unique()
    for approach in vool_approaches.keys():
        approach_stats = vool_approaches[approach]
        approach_stats["room_id"] = approach_stats["scene_id"].apply(
            lambda s: int(s.split("_")[0].split("FloorPlan")[1])
        )
        cutoff_analysis = approach_stats.groupby("cutoff")[[metric]].mean()
        best_cutoff = cutoff_analysis[metric].idxmax()
        approach_stats[metric] = approach_stats[metric] * 100
        for relation in relations:
            if relation == "[pad]":
                continue
            df = approach_stats[approach_stats.cutoff == best_cutoff]
            df = df[df.spatial_relation_name == relation]

            novel_vocab_mask = df.target_obj_name.isin(
                synonyms.values()
            ) | df.reference_obj_name.isin(synonyms.values())
            novel_class_mask = df.target_obj_name.isin(
                test_objs
            ) | df.reference_obj_name.isin(test_objs)

            vool_stats["approach"].append(approach)
            vool_stats["relation"].append(relation)
            novel_rooms_df = df[(df.split == "unseen_instances") & (~novel_class_mask)]
            mean_per_room = np.array(novel_rooms_df.groupby("room_id")[metric].mean())
            vool_stats["novel rooms"].append(np.nanmean(mean_per_room))
            novel_rooms_dr_df = df[
                (df.split == "unseen_instances_dr") & (~novel_class_mask)
            ]
            mean_per_room = np.array(
                novel_rooms_dr_df.groupby("room_id")[metric].mean()
            )
            vool_stats["novel visual"].append(np.nanmean(mean_per_room))

            unseen_class_df = df[novel_class_mask]
            vool_stats["novel class"].append(np.nanmean(unseen_class_df[metric]))
            unseen_vocab_df = df[
                (df.split == "unseen_instances_synonyms") & novel_vocab_mask
            ]
            vool_stats["novel vocab"].append(np.nanmean(unseen_vocab_df[metric]))
    vool_stats = pd.DataFrame.from_dict(vool_stats)
    for approach_i, approach in enumerate(vool_approaches.keys()):
        mean_df = pd.DataFrame.from_dict(
            {
                "approach": [approach],
                "relation": ["mean"],
                **{
                    split: [
                        np.array(
                            vool_stats[(vool_stats.approach == approach)][[split]]
                        ).mean()
                    ]
                    for split in [
                        "novel rooms",
                        "novel visual",
                        "novel vocab",
                        "novel class",
                    ]
                },
            }
        )
        vool_stats = pd.concat(
            [
                vool_stats.iloc[0 : (approach_i + 1) * 6 + approach_i],
                mean_df,
                vool_stats.iloc[(approach_i + 1) * 6 + approach_i :],
            ]
        )
    table = Table(title="FULL VOOL THOR", box=rich.box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Approach", justify="left")
    table.add_column("Spatial Relation", justify="left")
    table.add_column("Novel Room", justify="right")
    table.add_column("Novel Visual", justify="right")
    table.add_column("Novel Vocab", justify="right")
    table.add_column("Novel Class", justify="right")
    last_approach = ""
    for row in vool_stats.to_csv().split("\n")[1:-1]:
        (
            approach,
            spatial_relation,
            novel_room,
            novel_visual,
            novel_vocab,
            novel_class,
        ) = row.split(",")[1:]
        table.add_row(
            approach if approach != last_approach else "",
            spatial_relation,
            f"{float(novel_room):.01f}",
            f"{float(novel_visual):.01f}",
            f"{float(novel_vocab):.01f}",
            f"{float(novel_class):.01f}",
            end_section=spatial_relation == "mean",
            style=("green" if approach == "Ours" else "white"),
        )
        last_approach = approach
    console = Console()
    console.print(table)


def summarize_nyuv2(metric="voxel60x60x60_iou"):
    ssc_approaches = {
        "Ours (Supervised)": pickle.load(
            open(
                "models/ours/ovssc/ovssc_eval_stats_supervised_nyu_merged.pkl",
                "rb",
            )
        ),
        "Ours (Zeroshot)": pickle.load(
            open(
                "models/ours/ovssc/ovssc_eval_stats_zs_nyu_merged.pkl",
                "rb",
            )
        ),
    }
    classes = [
        "ceiling",
        "floor",
        "wall",
        "window",
        "chair",
        "bed",
        "sofa",
        "table",
        "tvs",
        "furn",
        "objs",
        "mean",
    ]
    table = Table(title="OVSSC NYU", box=rich.box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Approach", justify="left")
    for c in classes:
        table.add_column(c.title(), justify="right")
    for approach, approach_stats in ssc_approaches.items():
        approach_stats[metric] = approach_stats[metric] * 100
        cutoff_analysis = approach_stats.groupby("cutoff")[[metric]].mean()
        best_cutoff = cutoff_analysis[metric].idxmax()
        df = approach_stats[approach_stats.cutoff == best_cutoff]
        row = [approach]
        for c in classes:
            if c != "mean":
                row.append(f"{df[df.label == c][metric].mean():.01f}")
            else:
                row.append(
                    f'{np.array(df.groupby("label")[metric].mean()).mean():.01f}'
                )
        table.add_row(
            *row,
            end_section=approach == "Ours (Supervised)",
            style="green" if approach == "Ours (Zeroshot)" else "white",
        )
    console = Console()
    console.print(table)


if __name__ == "__main__":
    summarize_ovssc()
    summarize_vool()
    summarize_nyuv2()
