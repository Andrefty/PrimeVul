import json
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import defaultdict
import os

def parse_prediction(response_text, strategy):
    """
    Parses the model's textual response to a binary prediction.
    Returns 1 if predicted vulnerable, 0 if not vulnerable, None if unparseable.
    Handles responses with a <think>...</think> block. If content after </think>
    is empty or missing, it issues a warning and attempts to parse the whole response.
    Parses the final answer based on patterns like "(1) YES" or "(2) NO", 
    with fallbacks to "yes" or "no".
    """
    if response_text is None or response_text == "ERROR_NO_RESPONSE":
        return None

    normalized_response = response_text.lower().strip()
    
    final_answer_part = normalized_response # Default to whole response if no think tags
    think_tag_end = "</think>"
    
    if think_tag_end in normalized_response:
        parts = normalized_response.split(think_tag_end, 1)
        if len(parts) > 1:
            final_answer_part = parts[1].strip() # Text after </think>
            # print(f"Final answer part after </think>: '{final_answer_part}'")
        else:
            # This case implies </think> might be at the very end or split failed.
            print(f"No end think: {final_answer_part[:-70]}") # Print the last 70 characters of the response
        #     final_answer_part = "" # No clear answer part after think tag

    # If final_answer_part is empty after stripping (e.g. only whitespace after </think> or no answer after it)
    if not final_answer_part:
        return None

    # Parse the final_answer_part.
    # Check for patterns matching utils.py prompts like "(1) YES: ..." or "(2) NO: ..."
    if "(1) yes" in final_answer_part:
        return 1
    elif "(2) no" in final_answer_part:
        return 0
    # Fallback to simpler "yes" or "no" if the model's final answer is just that.
    # This order ensures that "(1) yes" isn't missed if it also contains "yes".
    elif "yes" in final_answer_part:
        return 1
    elif "no" in final_answer_part:
        return 0
    
    return None # Unparseable if none of the expected patterns are found

def main():
    parser = argparse.ArgumentParser(description="Evaluate model output from run_prompting.py and save all metrics.")
    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to the .jsonl output file from run_prompting.py")
    parser.add_argument('--prompt_strategy', type=str, required=True, choices=["std_cls", "cot"],
                        help="Prompt strategy used (std_cls or cot) for parsing responses.")
    parser.add_argument('--output_dir', type=str, default=".",
                        help="Directory to save the predictions.txt and metrics JSON file.")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize metrics dictionary
    base_input_filename = os.path.basename(args.input_file)
    metrics_filename = f"metrics_{os.path.splitext(base_input_filename)[0]}.json"
    metrics_filepath = os.path.join(args.output_dir, metrics_filename)

    metrics_output = {
        "input_file": args.input_file,
        "prompt_strategy": args.prompt_strategy,
        "total_samples_initially_read": 0,
        "samples_with_missing_target": 0,
        "unparseable_responses": 0,
        "valid_samples_for_std_metrics": 0, # Samples with valid target and parseable prediction
        "standard_metrics": {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
        },
        "pairwise_metrics": {
            "total_commit_ids_processed_for_pairing": 0,
            "total_structurally_valid_pairs_found": 0, # Pairs with target 0/1 by commit_id
            "pairs_skipped_due_to_unparseable_predictions": 0, # Structurally valid pairs where at least one prediction was None
            "pairs_used_for_px_calculation": 0, # Pairs where both predictions were parseable
            "P-C_count": 0, "P-C_ratio": 0.0,
            "P-V_count": 0, "P-V_ratio": 0.0,
            "P-B_count": 0, "P-B_ratio": 0.0,
            "P-R_count": 0, "P-R_ratio": 0.0,
        }
    }

    predictions_for_calc_vd_score = [] # For predictions.txt, maintains input order
    
    true_labels_for_std_metrics = []
    predicted_labels_for_std_metrics = []
    
    samples_by_commit_id = defaultdict(list) # For pair-wise metrics, keyed by commit_id
    
    num_unparseable = 0
    samples_read = 0
    missing_target_count = 0
    
    with open(args.input_file, 'r') as f:
        for line_idx, line in enumerate(f):
            samples_read += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line ({line_idx+1}): {line.strip()}")
                predictions_for_calc_vd_score.append(0.5) # Placeholder for calc_vd_score
                continue

            response_text = data.get("response")
            target = data.get("target")
            commit_id = data.get("commit_id") # Expected from run_prompting.py output
            sample_key = data.get("sample_key", f"line_{line_idx+1}")

            if target is None:
                print(f"Skipping sample {sample_key} due to missing target.")
                predictions_for_calc_vd_score.append(0.5) # Neutral placeholder
                missing_target_count += 1
                continue

            predicted_class = parse_prediction(response_text, args.prompt_strategy)

            # Populate predictions_for_calc_vd_score (maintains order)
            if predicted_class is None:
                predictions_for_calc_vd_score.append(0.5) # Placeholder for unparseable
            else:
                predictions_for_calc_vd_score.append(float(predicted_class))

            # Handle unparseable predictions for metrics
            if predicted_class is None:
                num_unparseable += 1
                # Still add to samples_by_commit_id with predicted=None for pair-wise accounting
                if commit_id:
                    samples_by_commit_id[commit_id].append({
                        "target": target, "predicted": None, 
                        "response": response_text, "sample_key": sample_key
                    })
                continue # Skip for standard F1/P/R/A if unparseable

            # For standard metrics (only if parseable)
            true_labels_for_std_metrics.append(target)
            predicted_labels_for_std_metrics.append(predicted_class)

            # For pair-wise metrics (add if commit_id exists)
            if commit_id:
                samples_by_commit_id[commit_id].append({
                    "target": target, "predicted": predicted_class, 
                    "sample_key": sample_key
                })

    metrics_output["total_samples_initially_read"] = samples_read
    metrics_output["samples_with_missing_target"] = missing_target_count
    metrics_output["unparseable_responses"] = num_unparseable
    metrics_output["valid_samples_for_std_metrics"] = len(true_labels_for_std_metrics)

    print(f"Processed {samples_read} total entries from input file.")
    if missing_target_count > 0:
        print(f"Warning: {missing_target_count} samples were skipped due to missing target values.")
    if num_unparseable > 0:
        print(f"Warning: {num_unparseable} responses were unparseable using the '{args.prompt_strategy}' strategy.")
        print(f"These {num_unparseable} unparseable responses are excluded from F1/Precision/Recall/Accuracy calculations.")
        print(f"For predictions.txt (for calc_vd_score.py), unparseable responses are represented as 0.5.")

    if not true_labels_for_std_metrics or not predicted_labels_for_std_metrics:
        print("\\n--- Standard Metrics ---")
        print("No valid (parseable) predictions found to calculate standard metrics.")
    else:
        f1 = f1_score(true_labels_for_std_metrics, predicted_labels_for_std_metrics, zero_division=0)
        precision = precision_score(true_labels_for_std_metrics, predicted_labels_for_std_metrics, zero_division=0)
        recall = recall_score(true_labels_for_std_metrics, predicted_labels_for_std_metrics, zero_division=0)
        accuracy = accuracy_score(true_labels_for_std_metrics, predicted_labels_for_std_metrics)

        metrics_output["standard_metrics"]["accuracy"] = accuracy
        metrics_output["standard_metrics"]["precision"] = precision
        metrics_output["standard_metrics"]["recall"] = recall
        metrics_output["standard_metrics"]["f1_score"] = f1

        print("\\n--- Standard Metrics ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    # Write predictions.txt for calc_vd_score.py
    pred_file_path = os.path.join(args.output_dir, "predictions.txt")
    with open(pred_file_path, 'w') as pf:
        for p_val in predictions_for_calc_vd_score:
            pf.write(f"{p_val}\\n")
    print(f"\\nPredictions for VD-Score saved to: {pred_file_path}")
    print(f"You can now run: python <path_to_calc_vd_score.py> --pred_file {pred_file_path} --test_file <PATH_TO_ORIGINAL_TEST_FILE.jsonl>")

    # Calculate Pair-wise metrics using commit_id
    pc_count, pv_count, pb_count, pr_count = 0, 0, 0, 0
    structurally_valid_pairs = 0
    skipped_pairs_due_to_unparseable = 0
    
    metrics_output["pairwise_metrics"]["total_commit_ids_processed_for_pairing"] = len(samples_by_commit_id)

    for commit_id_key, items in samples_by_commit_id.items():
        if len(items) == 2:
            item1, item2 = items[0], items[1]
            
            # Check for valid vuln(1)/benign(0) pair structure
            if not ((item1['target'] == 1 and item2['target'] == 0) or \
                    (item1['target'] == 0 and item2['target'] == 1)):
                # This commit_id does not represent a valid 0/1 pair.
                continue 

            structurally_valid_pairs += 1
            
            vuln_sample = item1 if item1['target'] == 1 else item2
            benign_sample = item2 if item1['target'] == 1 else item1
            
            pred_vuln = vuln_sample['predicted']
            pred_benign = benign_sample['predicted']

            if pred_vuln is None or pred_benign is None:
                skipped_pairs_due_to_unparseable +=1
                continue # Skip P-X calculation if either prediction is unparseable

            # Both predictions are parseable, proceed with P-X classification
            if pred_vuln == 1 and pred_benign == 0: pc_count += 1
            elif pred_vuln == 1 and pred_benign == 1: pv_count += 1
            elif pred_vuln == 0 and pred_benign == 0: pb_count += 1
            elif pred_vuln == 0 and pred_benign == 1: pr_count += 1
        else:
            # Optional: Log if a commit_id has != 2 items
            print(f"Commit ID {commit_id_key} does not have exactly two items for pairing. Count: {len(items)}.")

    metrics_output["pairwise_metrics"]["total_structurally_valid_pairs_found"] = structurally_valid_pairs
    metrics_output["pairwise_metrics"]["pairs_skipped_due_to_unparseable_predictions"] = skipped_pairs_due_to_unparseable
    
    pairs_used_for_px_calc = structurally_valid_pairs - skipped_pairs_due_to_unparseable
    metrics_output["pairwise_metrics"]["pairs_used_for_px_calculation"] = pairs_used_for_px_calc

    if pairs_used_for_px_calc > 0:
        metrics_output["pairwise_metrics"]["P-C_count"] = pc_count
        metrics_output["pairwise_metrics"]["P-C_ratio"] = pc_count / pairs_used_for_px_calc
        metrics_output["pairwise_metrics"]["P-V_count"] = pv_count
        metrics_output["pairwise_metrics"]["P-V_ratio"] = pv_count / pairs_used_for_px_calc
        metrics_output["pairwise_metrics"]["P-B_count"] = pb_count
        metrics_output["pairwise_metrics"]["P-B_ratio"] = pb_count / pairs_used_for_px_calc
        metrics_output["pairwise_metrics"]["P-R_count"] = pr_count
        metrics_output["pairwise_metrics"]["P-R_ratio"] = pr_count / pairs_used_for_px_calc
        
        print("\\n--- Pair-wise Metrics ---")
        print(f"Total commit_ids processed for pairing: {len(samples_by_commit_id)}")
        print(f"Total structurally valid pairs (target 0/1): {structurally_valid_pairs}")
        print(f"Pairs skipped due to one/both predictions being unparseable: {skipped_pairs_due_to_unparseable}")
        print(f"Pairs used for P-X calculations (both predictions parseable): {pairs_used_for_px_calc}")

        print(f"P-C (Correct - Vuln:1, Benign:0): {pc_count} ({metrics_output['pairwise_metrics']['P-C_ratio']:.4f})")
        print(f"P-V (Both Vuln - Vuln:1, Benign:1): {pv_count} ({metrics_output['pairwise_metrics']['P-V_ratio']:.4f})")
        print(f"P-B (Both Benign - Vuln:0, Benign:0): {pb_count} ({metrics_output['pairwise_metrics']['P-B_ratio']:.4f})")
        print(f"P-R (Reversed - Vuln:0, Benign:1): {pr_count} ({metrics_output['pairwise_metrics']['P-R_ratio']:.4f})")
        
        sum_px_counts = pc_count + pv_count + pb_count + pr_count
        if sum_px_counts != pairs_used_for_px_calc:
            print(f"Warning: Sum of P-X counts ({sum_px_counts}) does not match number of pairs used for P-X calculations ({pairs_used_for_px_calc}). This indicates an issue in P-X logic or unhandled prediction values.")
    else:
        print("\\n--- Pair-wise Metrics ---")
        print(f"Total commit_ids processed for pairing: {len(samples_by_commit_id)}")
        print(f"Total structurally valid pairs (target 0/1): {structurally_valid_pairs}")
        if structurally_valid_pairs > 0 :
             print(f"Pairs skipped due to one/both predictions being unparseable: {skipped_pairs_due_to_unparseable}")
        print("No pairs with both predictions parseable were found for P-X metric calculations.")

    # Save all metrics to the JSON file
    try:
        with open(metrics_filepath, 'w') as mf:
            json.dump(metrics_output, mf, indent=4)
        print(f"\\nEvaluation metrics saved to: {metrics_filepath}")
    except Exception as e:
        print(f"\\nError saving metrics to JSON: {e}")

if __name__ == "__main__":
    main()
