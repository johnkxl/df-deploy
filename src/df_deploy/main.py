from tqdm import tqdm

from df_deploy.saving import save_json
from df_deploy.loading import load_df
from df_deploy.preprocessing import DataProcessor
from df_deploy.cli import setup_program
from df_deploy.models import DeploymentModel
from df_deploy.model_loader import ModelLoader, base_performances
from df_deploy.predicting.utils import compute_performance_deltas, predict_evaluate


def main() -> None:
    
    opts = setup_program()

    processor = DataProcessor.from_X_prepared_cat_cont(
        X=opts.X,
        X_cat=opts.X_cat,
        X_cont=opts.X_cont,
        is_classification=opts.is_classification,
        nan_strategy=opts.nan_handling,
        labels=opts.labels,
        scaler=opts.scaler,
    )
    processor.save_all(opts.program_dir / "processor")

    # if opts.extra_params:
    #     print(f"Extra hyperparameters: {opts.extra_params}")

    save_json(opts.feature_selections, opts.program_dir / "feature_selections.json")

    print(opts.loading_str, end="", flush=True)
    loader = ModelLoader(
        performance_table=opts.final_performances,
        tuned_models=opts.tuned_models,
        feature_selections=opts.feature_selections,
        is_classification=opts.is_classification,
    )
    models: list[DeploymentModel] = loader.load_top_models(
        opts.metric, opts.ranking_method, opts.top, # opts.extra_params
    )
    print("done.")

    models_info = {}

    progress_bar = tqdm(enumerate(models), total=len(models), desc="Training")
    
    for i, model in progress_bar:
        progress_bar.set_description(f"Training {model.fullname}")
        
        model.fit(opts.X, opts.y)

        model_pkl_file = f"{model.fullname}.pkl"
        model.save_to_file(opts.models_dir / model_pkl_file)

        model_metrics = base_performances(
            model, opts.final_performances, opts.ranking_method,
        )

        model_info = {
            "model": model.shortname,
            "rank": i + 1,
            "file": model_pkl_file,
            "feature_selection": model.feature_selection,
            "hyperparameters": model.full_args,
            opts.ranking_method: model_metrics,
        }
        models_info[model.fullname] = model_info

    save_json(models_info, opts.models_dir / "models_info.json")
    save_json({"target": opts.target}, opts.program_dir / "meta.json")
    
    if opts.holdout_path is None:
        return
    
    holdout_df = load_df(opts.holdout_path)
    # Drop any rows where the target column is null
    holdout_df = holdout_df.dropna(subset=[opts.target])

    predictions_dir = opts.program_dir / "predictions"
    model_evaluator = predict_evaluate(
        holdout_df,
        opts.target,
        models,
        processor,
        predictions_dir,
        opts.embedding_model_path,
    )
    performance_deltas = compute_performance_deltas(
        model_evaluator,
        models_info,
        opts.ranking_method,
    )
    save_json(performance_deltas, predictions_dir / "performance_deltas.json")

    return