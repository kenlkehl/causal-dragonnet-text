"""Command-line interface for OCI experiments."""

import argparse
import sys
from pathlib import Path
import logging

from .config import ExperimentConfig, create_default_config
from .experiments.runner import ExperimentRunner
from .utils.system import setup_logging, limit_threads


def main():
    """Main entry point for OCI CLI."""
    limit_threads(n_threads=1)
    
    parser = argparse.ArgumentParser(
        description="Oncology Causal Inference: Causal inference from clinical text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default config
  oci init --output config.json

  # Run experiment with config
  oci run --config config.json

  # Run with custom settings
  oci run --config config.json --device cuda:0 --workers 4
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    init_parser = subparsers.add_parser('init', help='Create default configuration file')
    init_parser.add_argument(
        '--output', '-o',
        default='oci_config.json',
        help='Output path for config file (default: oci_config.json)'
    )
    
    run_parser = subparsers.add_parser('run', help='Run experiment from config')
    run_parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to configuration JSON file'
    )
    run_parser.add_argument(
        '--device',
        help='Override device from config (e.g., cuda:0, mps, cpu)'
    )
    run_parser.add_argument(
        '--workers',
        type=int,
        help='Override number of workers from config'
    )
    run_parser.add_argument(
        '--output-dir',
        help='Override output directory from config'
    )
    run_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    run_parser.add_argument(
        '--skip-pretraining',
        action='store_true',
        help='Skip pretraining even if enabled in config'
    )
    run_parser.add_argument(
        '--r-stage-min-propensity',
        type=float,
        help=(
            'Override minimum nuisance propensity score eligible for '
            'agentic_attention_variable_forest R-stage training'
        )
    )
    run_parser.add_argument(
        '--r-stage-max-propensity',
        type=float,
        help=(
            'Override maximum nuisance propensity score eligible for '
            'agentic_attention_variable_forest R-stage training'
        )
    )
    run_parser.add_argument(
        '--inner-fold-parallelism',
        help=(
            "Override inner cross-fit fold parallelism for "
            "agentic_attention_variable_forest or causal_forest R-learner runs. "
            "Use 'auto' or a positive integer."
        )
    )
    run_parser.add_argument(
        '--effect-objective',
        choices=['squared_r_loss', 'logistic_r_loss', 'pseudo_outcome_mse'],
        help=(
            "Override neural effect-stage objective for "
            "model_type='agentic_attention_variable_forest'."
        )
    )
    run_parser.add_argument(
        '--neural-stage-mode',
        choices=['staged', 'joint_rlearner', 'interaction_outcome', 'tarnet_offset'],
        help=(
            "Override neural learning mode for "
            "model_type='agentic_attention_variable_forest'."
        )
    )
    run_parser.add_argument(
        '--joint-rlearner-gamma',
        type=float,
        help=(
            "Override detached-nuisance R-loss weight for "
            "agentic_attention_variable_forest joint_rlearner mode."
        )
    )
    run_parser.add_argument(
        '--interaction-l2-weight',
        type=float,
        help=(
            "Override interaction/offset component L2 penalty for "
            "agentic_attention_variable_forest interaction_outcome or "
            "tarnet_offset mode."
        )
    )
    run_parser.add_argument(
        '--tarnet-offset-batch-size',
        type=int,
        help=(
            "Override TarNet-offset batch size for "
            "agentic_attention_variable_forest tarnet_offset mode."
        )
    )
    run_parser.add_argument(
        '--tarnet-offset-heterogeneity-weight',
        type=float,
        help=(
            "Override weight for the TarNet-offset within-batch "
            "heterogeneity floor."
        )
    )
    run_parser.add_argument(
        '--tarnet-offset-min-logit-std',
        type=float,
        help=(
            "Override target minimum within-batch std of offset1-offset0 "
            "in TarNet-offset mode."
        )
    )
    run_parser.add_argument(
        '--alpha-propensity',
        type=float,
        help="Override applied inference treatment/propensity loss weight."
    )
    
    args = parser.parse_args()
    
    if args.command == 'init':
        create_default_config(args.output)
        print(f"\nEdit {args.output} and then run:")
        print(f"  oci run --config {args.output}")
        return 0
    
    elif args.command == 'run':
        level = logging.DEBUG if args.verbose else logging.INFO
        setup_logging(level=level)
        
        try:
            config = ExperimentConfig.from_json(args.config)
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1
        
        if args.device:
            config.device = args.device
        if args.workers:
            config.num_workers = args.workers
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.skip_pretraining:
            config.pretraining.enabled = False
        if args.r_stage_min_propensity is not None or args.r_stage_max_propensity is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "agentic_attention_variable_forest":
                print(
                    "--r-stage-min-propensity/--r-stage-max-propensity only apply "
                    "to model_type='agentic_attention_variable_forest'"
                )
                return 1
            avf_config = (
                config.applied_inference.architecture
                .agentic_attention_variable_forest
            )
            if args.r_stage_min_propensity is not None:
                avf_config.r_stage_min_propensity = args.r_stage_min_propensity
            if args.r_stage_max_propensity is not None:
                avf_config.r_stage_max_propensity = args.r_stage_max_propensity
            if not (
                0.0
                <= avf_config.r_stage_min_propensity
                < avf_config.r_stage_max_propensity
                <= 1.0
            ):
                print(
                    "R-stage propensity bounds must satisfy 0 <= min < max <= 1"
                )
                return 1
        if args.inner_fold_parallelism is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type == "agentic_attention_variable_forest":
                avf_config = (
                    config.applied_inference.architecture
                    .agentic_attention_variable_forest
                )
                avf_config.fold_parallelism = str(args.inner_fold_parallelism)
            elif model_type == "causal_forest":
                cf_config = config.applied_inference.architecture.causal_forest
                cf_config.rlearner_inner_fold_parallelism = str(args.inner_fold_parallelism)
            else:
                print(
                    "--inner-fold-parallelism only applies to "
                    "model_type='agentic_attention_variable_forest' or "
                    "model_type='causal_forest'"
                )
                return 1
        if args.effect_objective is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "agentic_attention_variable_forest":
                print(
                    "--effect-objective only applies to "
                    "model_type='agentic_attention_variable_forest'"
                )
                return 1
            (
                config.applied_inference.architecture
                .agentic_attention_variable_forest
                .effect_objective
            ) = args.effect_objective
        if args.neural_stage_mode is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "agentic_attention_variable_forest":
                print(
                    "--neural-stage-mode only applies to "
                    "model_type='agentic_attention_variable_forest'"
                )
                return 1
            (
                config.applied_inference.architecture
                .agentic_attention_variable_forest
                .neural_stage_mode
            ) = args.neural_stage_mode
        if args.joint_rlearner_gamma is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "agentic_attention_variable_forest":
                print(
                    "--joint-rlearner-gamma only applies to "
                    "model_type='agentic_attention_variable_forest'"
                )
                return 1
            (
                config.applied_inference.architecture
                .agentic_attention_variable_forest
                .joint_rlearner_gamma
            ) = args.joint_rlearner_gamma
        if args.interaction_l2_weight is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "agentic_attention_variable_forest":
                print(
                    "--interaction-l2-weight only applies to "
                    "model_type='agentic_attention_variable_forest'"
                )
                return 1
            (
                config.applied_inference.architecture
                .agentic_attention_variable_forest
                .interaction_l2_weight
            ) = args.interaction_l2_weight
        if args.tarnet_offset_batch_size is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "agentic_attention_variable_forest":
                print(
                    "--tarnet-offset-batch-size only applies to "
                    "model_type='agentic_attention_variable_forest'"
                )
                return 1
            (
                config.applied_inference.architecture
                .agentic_attention_variable_forest
                .tarnet_offset_batch_size
            ) = args.tarnet_offset_batch_size
        if args.tarnet_offset_heterogeneity_weight is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "agentic_attention_variable_forest":
                print(
                    "--tarnet-offset-heterogeneity-weight only applies to "
                    "model_type='agentic_attention_variable_forest'"
                )
                return 1
            (
                config.applied_inference.architecture
                .agentic_attention_variable_forest
                .tarnet_offset_heterogeneity_weight
            ) = args.tarnet_offset_heterogeneity_weight
        if args.tarnet_offset_min_logit_std is not None:
            model_type = getattr(config.applied_inference.architecture, 'model_type', None)
            if model_type != "agentic_attention_variable_forest":
                print(
                    "--tarnet-offset-min-logit-std only applies to "
                    "model_type='agentic_attention_variable_forest'"
                )
                return 1
            (
                config.applied_inference.architecture
                .agentic_attention_variable_forest
                .tarnet_offset_min_logit_std
            ) = args.tarnet_offset_min_logit_std
        if args.alpha_propensity is not None:
            config.applied_inference.training.alpha_propensity = args.alpha_propensity
        
        try:
            config.validate()
        except ValueError as e:
            print(f"Configuration error: {e}")
            return 1
        
        runner = ExperimentRunner(config)
        
        try:
            results = runner.run()
            print(f"\n{'='*80}")
            print("EXPERIMENT COMPLETE")
            print(f"{'='*80}")
            print(f"Results saved to: {config.output_dir}")
            
            if results.get('applied_inference'):
                print(f"\nApplied inference results: {results['applied_inference']}")
            
            return 0
            
        except Exception as e:
            logging.error(f"Experiment failed: {e}", exc_info=True)
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
