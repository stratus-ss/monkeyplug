import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Set

def load_swear_words(swear_file_path: str) -> Set[str]:
    """
    Load swear words from JSON file.
    Handles various JSON formats (array, object with words key, etc.)
    """
    with open(swear_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        swear_words = set(word.lower().strip() for word in data)
    elif isinstance(data, dict):
        # Common keys: 'words', 'profanity', 'swears', 'list'
        for key in ['words', 'profanity', 'swears', 'list', 'items']:
            if key in data:
                swear_words = set(word.lower().strip() for word in data[key])
                break
        else:
            # If no recognized key, use all values
            swear_words = set(str(v).lower().strip() for v in data.values())
    else:
        raise ValueError(f"Unsupported JSON format in {swear_file_path}")
    
    return swear_words

def find_profanity_in_transcript(transcript_data: List[Dict], swear_words: Set[str], 
                                confidence_threshold: float = 0.65) -> tuple:
    """
    Find all profanity instances in a transcript.
    Returns tuple of (all_matches, censored_matches) where censored_matches meet confidence threshold.
    
    Args:
        transcript_data: List of word dictionaries from transcript
        swear_words: Set of profanity words to match against
        confidence_threshold: Minimum confidence (0.0-1.0) to actually censor (default: 0.65)
    
    Returns:
        tuple: (all_profanity_matches, censored_matches)
    """
    all_matches = []
    censored_matches = []
    
    for word_data in transcript_data:
        word = word_data.get('word', '').lower().strip()
        # Remove common punctuation for matching
        word_clean = word.strip('.,!?;:\'"')
        
        if word_clean in swear_words:
            confidence = word_data.get('conf', 1.0)
            match = {
                'word': word_data.get('word'),
                'word_clean': word_clean,
                'start': word_data.get('start'),
                'end': word_data.get('end'),
                'confidence': confidence,
            }
            all_matches.append(match)
            
            # Only mark for censoring if confidence meets threshold
            if confidence >= confidence_threshold:
                censored_matches.append(match)
                word_data['scrub'] = True
            else:
                word_data['scrub'] = False
    
    return all_matches, censored_matches

def process_transcript_file(transcript_path: str, swear_words: Set[str], 
                           confidence_threshold: float = 0.65) -> Dict:
    """
    Process a single transcript file and return profanity report.
    
    Args:
        transcript_path: Path to transcript JSON file
        swear_words: Set of profanity words
        confidence_threshold: Minimum confidence to censor (default: 0.65)
    
    Returns:
        Dict with profanity analysis results
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    all_matches, censored_matches = find_profanity_in_transcript(
        transcript_data, swear_words, confidence_threshold
    )
    
    return {
        'file': transcript_path,
        'total_words': len(transcript_data),
        'profanity_found': len(all_matches),
        'profanity_censored': len(censored_matches),
        'profanity_filtered': len(all_matches) - len(censored_matches),
        'all_matches': all_matches,
        'censored_matches': censored_matches
    }

def process_all_transcripts(transcript_dir: str, swear_file: str, 
                           confidence_threshold: float = 0.65, output_file: str = None):
    """
    Process all transcript JSON files in a directory.
    
    Args:
        transcript_dir: Directory containing transcript JSON files
        swear_file: Path to swear words JSON file
        confidence_threshold: Minimum confidence to censor (default: 0.65)
        output_file: Optional output file for JSON report
    """
    # Load swear words
    swear_words = load_swear_words(swear_file)
    print(f"Loaded {len(swear_words)} swear words from {swear_file}")
    print(f"Using confidence threshold: {confidence_threshold} (only words with confidence >= {confidence_threshold} will be censored)")
    
    # Find all transcript JSON files
    transcript_dir_path = Path(transcript_dir)
    transcript_files = sorted(transcript_dir_path.glob('*.json'))
    
    if not transcript_files:
        print(f"No JSON files found in {transcript_dir}")
        return
    
    print(f"Found {len(transcript_files)} transcript files")
    
    # Process each transcript
    results = []
    total_found = 0
    total_censored = 0
    total_filtered = 0
    
    for transcript_file in transcript_files:
        print(f"\nProcessing: {transcript_file.name}")
        result = process_transcript_file(str(transcript_file), swear_words, confidence_threshold)
        results.append(result)
        total_found += result['profanity_found']
        total_censored += result['profanity_censored']
        total_filtered += result['profanity_filtered']
        
        print(f"  Found {result['profanity_found']} instances of profanity")
        print(f"  Would censor {result['profanity_censored']} (confidence >= {confidence_threshold})")
        if result['profanity_filtered'] > 0:
            print(f"  Filtered out {result['profanity_filtered']} (low confidence)")
        
        # Print censored matches
        if result['censored_matches']:
            print("  Censored matches:")
            for match in result['censored_matches']:
                print(f"    - '{match['word']}' at {match['start']:.2f}s - {match['end']:.2f}s (conf: {match['confidence']:.3f})")
        
        # Print filtered matches
        if result['profanity_filtered'] > 0:
            filtered = [m for m in result['all_matches'] if m['confidence'] < confidence_threshold]
            print("  Filtered (low confidence):")
            for match in filtered[:5]:  # Show first 5
                print(f"    - '{match['word']}' at {match['start']:.2f}s - {match['end']:.2f}s (conf: {match['confidence']:.3f})")
            if len(filtered) > 5:
                print(f"    ... and {len(filtered) - 5} more")
    
    # Generate summary report
    summary = {
        'total_transcripts': len(transcript_files),
        'confidence_threshold': confidence_threshold,
        'total_profanity_found': total_found,
        'total_profanity_censored': total_censored,
        'total_profanity_filtered': total_filtered,
        'swear_words_loaded': len(swear_words),
        'results': results
    }
    
    # Save results to output file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull report saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total transcripts processed: {summary['total_transcripts']}")
    print(f"Total profanity found: {summary['total_profanity_found']}")
    print(f"Total that would be censored: {summary['total_profanity_censored']} (confidence >= {confidence_threshold})")
    print(f"Total filtered out: {summary['total_profanity_filtered']} (low confidence)")
    print(f"Swear words in dictionary: {len(swear_words)}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # # Generate FFmpeg filter commands
    # print(f"\n{'='*60}")
    # print(f"FFMPEG MUTE COMMANDS")
    # print(f"{'='*60}")
    
    # for result in results:
    #     if result['profanity_count'] > 0:
    #         audio_file = Path(result['file']).stem + '.m4b'  # Adjust extension as needed
    #         print(f"\n# {audio_file}")
            
    #         # Generate volume filter for muting
    #         filter_parts = []
    #         for match in result['matches']:
    #             start = match['start'] - 0.1  # Add padding before
    #             end = match['end'] + 0.1      # Add padding after
    #             filter_parts.append(f"volume=0:enable='between(t,{start:.2f},{end:.2f})'")
            
    #         filter_chain = ",".join(filter_parts)
            
#            print(f"ffmpeg -i {audio_file} -af \"{filter_chain}\" -c:a copy {Path(audio_file).stem}_clean.m4b")
    
    return summary

def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description='Find profanity in transcript JSON files and generate a report.'
    )
    
    parser.add_argument(
        '--swear-words-file',
        type=str,
        default='swear_list.json',
        help='Path to the JSON file containing swear words (default: swear_list.json)'
    )
    
    parser.add_argument(
        '--transcript-dir',
        type=str,
        default='output',
        help='Directory containing transcript JSON files (default: output)'
    )
    
    parser.add_argument(
        '--output-report',
        type=str,
        default='profanity_report.json',
        help='Path for the output report JSON file (default: profanity_report.json)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.65,
        help='Minimum confidence level (0.0-1.0) required to censor a word (default: 0.65, matching monkeyplug default)'
    )
    
    args = parser.parse_args()
    
    # Process all transcripts and generate report
    summary = process_all_transcripts(
        transcript_dir=args.transcript_dir,
        swear_file=args.swear_words_file,
        confidence_threshold=args.confidence_threshold,
        output_file=args.output_report
    )
    
    return summary

# Main execution
if __name__ == "__main__":
    main()
