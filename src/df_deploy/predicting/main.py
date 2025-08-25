from df_deploy.models import DeploymentModel
from df_deploy.predicting.cli import get_options
from df_deploy.predicting.utils import predict_evaluate, compute_performance_deltas
from df_deploy.saving import save_json


def main() -> None:
    opts = get_options()

    model = DeploymentModel.from_file(opts.model_path)

    model_evaluator = predict_evaluate(
        df=opts.df,
        target=opts.target,
        models=[model],
        processor=opts.processor,
        out=opts.program_dir,
        embed_dir=opts.embed_dir,
    )
    performance_deltas = compute_performance_deltas(
        model_evaluator,
        opts.models_info,
        # opts.ranking_method,
        ranking_method="holdout",
    )
    save_json(performance_deltas, opts.program_dir / "performance_deltas.json")
    return