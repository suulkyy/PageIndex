import argparse
import os
import json
from pageindex import *
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader, configure_llm_runtime

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

    parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config.yaml)')
    parser.add_argument('--base-url', type=str, default=None,
                      help='Base URL for hosted vLLM/Ollama-compatible servers')
    parser.add_argument('--vllm-max-model-len', type=int, default=None,
                      help='Server max model length for vLLM/hosted_vllm')
    parser.add_argument('--vllm-max-tokens', type=int, default=None,
                      help='Per-request output cap for vLLM/hosted_vllm; 0 leaves server default')
    parser.add_argument('--vllm-timeout', type=float, default=None,
                      help='Request timeout for vLLM/hosted_vllm')
    parser.add_argument('--vllm-ctx-margin', type=int, default=None,
                      help='Context safety margin for vLLM max_tokens clamping')
    parser.add_argument('--llm-concurrency', type=int, default=None,
                      help='Max in-flight async LLM calls')
    parser.add_argument('--group-max-tokens', type=int, default=None,
                      help='Max tokens per no-TOC page group')
    parser.add_argument('--group-prompt-overhead', type=int, default=None,
                      help='Prompt reserve used when sizing no-TOC page groups')
    parser.add_argument('--toc-chunk-max-tokens', type=int, default=None,
                      help='Max tokens per chunk for LLM TOC transformation fallback')

    parser.add_argument('--toc-check-pages', type=int, default=None,
                      help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--toc-verify-sample', type=int, default=None,
                      help='Number of TOC entries to verify; 0 checks all (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=None,
                      help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=None,
                      help='Maximum number of tokens per node (PDF only)')

    parser.add_argument('--if-add-node-id', type=str, default=None,
                      help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default=None,
                      help='Whether to add summary to the node')
    parser.add_argument('--if-add-doc-description', type=str, default=None,
                      help='Whether to add doc description to the doc')
    parser.add_argument('--if-add-node-text', type=str, default=None,
                      help='Whether to add text to the node')
                      
    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    args = parser.parse_args()
    # Seed the per-process LLM runtime registry from CLI args. Must run
    # before any pageindex.* call so `_provider_kwargs`, `_resolve_max_tokens`,
    # `_get_llm_sem`, and `page_list_to_group_text` see the overrides on
    # their first read. None values fall through to env-var defaults.
    configure_llm_runtime(
        vllm_base_url=args.base_url,
        ollama_base_url=args.base_url,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_max_tokens=args.vllm_max_tokens,
        vllm_timeout=args.vllm_timeout,
        vllm_ctx_margin=args.vllm_ctx_margin,
        llm_concurrency=args.llm_concurrency,
        pageindex_group_max_tokens=args.group_max_tokens,
        pageindex_group_prompt_overhead=args.group_prompt_overhead,
        toc_chunk_max_tokens=args.toc_chunk_max_tokens,
    )
    
    # Validate that exactly one file type is specified
    if not args.pdf_path and not args.md_path:
        raise ValueError("Either --pdf_path or --md_path must be specified")
    if args.pdf_path and args.md_path:
        raise ValueError("Only one of --pdf_path or --md_path can be specified")
    
    if args.pdf_path:
        # Validate PDF file
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")
            
        # Process PDF file
        user_opt = {
            'model': args.model,
            'toc_check_page_num': args.toc_check_pages,
            'toc_verify_sample_num': args.toc_verify_sample,
            'max_page_num_each_node': args.max_pages_per_node,
            'max_token_num_each_node': args.max_tokens_per_node,
            'if_add_node_id': args.if_add_node_id,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
        }
        opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

        # Process the PDF
        toc_with_page_number = page_index_main(args.pdf_path, opt)
        print('Parsing done, saving to file...')
        
        # Save results
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]    
        output_dir = './results'
        output_file = f'{output_dir}/{pdf_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2)
        
        print(f'Tree structure saved to: {output_file}')
            
    elif args.md_path:
        # Validate Markdown file
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")
            
        # Process markdown file
        print('Processing markdown file...')
        
        # Process the markdown
        import asyncio
        
        # Use ConfigLoader to get consistent defaults (matching PDF behavior)
        from pageindex.utils import ConfigLoader
        config_loader = ConfigLoader()
        
        # Create options dict with user args
        user_opt = {
            'model': args.model,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
            'if_add_node_id': args.if_add_node_id
        }
        
        # Load config with defaults from config.yaml
        opt = config_loader.load(user_opt)
        
        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id
        ))
        
        print('Parsing done, saving to file...')
        
        # Save results
        md_name = os.path.splitext(os.path.basename(args.md_path))[0]    
        output_dir = './results'
        output_file = f'{output_dir}/{md_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')
