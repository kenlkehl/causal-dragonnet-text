# synthetic_data/cli.py
"""Command-line interface for synthetic data generation."""

import argparse
import logging
import sys

from .config import SyntheticDataConfig, LLMConfig, DEFAULT_CLINICAL_QUESTION
from .generator import generate_synthetic_dataset


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinical datasets with known causal structure using LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with local vLLM server
  python -m synthetic_data.cli --api-url http://localhost:8000/v1 --dataset-size 100

  # Generate with OpenAI API
  python -m synthetic_data.cli --api-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY --model gpt-4

  # Custom clinical question
  python -m synthetic_data.cli --clinical-question "Compare pembrolizumab with nivolumab for NSCLC" 
        """,
    )
    
    # Clinical question
    parser.add_argument(
        "--clinical-question",
        type=str,
        default=DEFAULT_CLINICAL_QUESTION,
        help="Comparative effectiveness research question",
    )
    
    # Dataset parameters
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=500,
        help="Number of patients to generate (default: 500)",
    )
    parser.add_argument(
        "--treatment-coefficient",
        type=float,
        default=1.0,
        help="Treatment coefficient in outcome equation on logit scale (default: 1.0)",
    )
    
    # LLM parameters
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:8000/v1 for vLLM)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key (can be blank for local models)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model name to use (default: 'openai/gpt-oss-120b')",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20000,
        help="Max tokens per LLM response (default: 20000)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synthetic_output",
        help="Output directory for generated files (default: ./synthetic_output)",
    )
    
    # Execution
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=40,
        help="Number of parallel workers for patient generation (default: 4)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Build configuration
    config = SyntheticDataConfig(
        clinical_question=args.clinical_question,
        dataset_size=args.dataset_size,
        treatment_coefficient=args.treatment_coefficient,
        output_dir=args.output_dir,
        seed=args.seed,
        llm=LLMConfig(
            api_base_url=args.api_url,
            api_key=args.api_key,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
    )
    
    # Run generation
    try:
        df, metadata = generate_synthetic_dataset(
            config=config,
            num_workers=args.num_workers,
            show_progress=True,
        )
        
        print(f"\n✓ Generated {len(df)} patients")
        print(f"  - Treatment rate: {df['treatment'].mean():.1%}")
        print(f"  - Outcome rate: {df['outcome'].mean():.1%}")
        print(f"  - Output: {args.output_dir}/dataset.parquet")
        print(f"  - Metadata: {args.output_dir}/metadata.json")
        
    except Exception as e:
        logging.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
