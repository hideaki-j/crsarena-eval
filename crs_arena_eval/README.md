# CRSArena-Eval Dataset

Our **CRSArena-Eval dataset** contains a collection of human-annotated conversations between users and 9 Conversational Recommender Systems (CRSs). Created on the [CRSArena-Dial dataset](https://github.com/iai-group/crsarena-dial) [1], CRSArena-Eval is designed to evaluate the CRS evaluators (meta-evaluation).

## Statistics

| Statistic                     | Value  |
|-------------------------------|--------|
| # Conversations                 | 467    |
| # Utterances                    | 4,473  |
| Avg. utterances per conversation | 9.58   |
| Avg. words per user utterance | 7.53   |
| Avg. words per system utterance | 15.18  |
| # CRSs                        | 9      |
| # Turn-level aspects            | 2      |
| # Dialogue-level aspects        | 5      |
| # Final labels (after aggregation) | 6,805  |

**CRSs**: CRSArena-Eval includes 9 CRSs: `chatgpt_redial`, `barcor_redial`, `unicrs_redial`, `crb-crs_redial`, `kbrd_redial`, `chatgpt_opendialkg`, `barcor_opendialkg`, `unicrs_opendialkg`, and `kbrd_opendialkg`. The CRS model is shown in `conv_id`. For the details of these CRSs, please refer to [Section 3 of our paper](https://arxiv.org/abs/2506.00314) or [Bernard et al. [1]](https://dl.acm.org/doi/10.1145/3701551.3704120).

<details>
<summary>Details: Final Label Statistics</summary>

Here, we provide the statistics of the final human labels (after aggregation).

| Aspect   | Turn- or Dialogue-level | # Annotations |
|---------|------------------------|---------------|
| Understanding | Dialogue-level         | 467           |
| Task Completion | Dialogue-level         | 467           |
| Interest Arousal | Dialogue-level         | 467           |
| Efficiency | Dialogue-level         | 467           |
| Overall Impression | Dialogue-level         | 467           |
| Relevance | Turn-level             | 2,235         |
| Interestingness | Turn-level             | 2,235         |
| **Total**   |                        | **6,805**     |

</details>




## Data Structure

The dataset is a single JSON file containing an list of conversation entries. Each object represents one complete dialogue and its associated annotations.

```python
[
  { # Conversation 1
    "conv_id": "...",
    "dialogue": [
      { "role": "USER", "utterance": "..." }, # User turn
      { "role": "ASST", "utterance": "...", "turn_level_aggregated": {} }, # Only assistant turns have annotations
      # ... more turns
    ],
    "dial_level_aggregated": { ... }
  } # ... more conversations
]
```

**Conversation Object**

Each conversation object in the list has three primary keys:

*   **`conv_id`**: A unique string that identifies the conversation.
*   **`dialogue`**: A list of items, where each item represents a single turn in the dialogue.
*   **`dial_level_aggregated`**: A dict structure containing aggregated human scores that evaluate the overall quality of the assistant's performance throughout the conversation. Note that each annotation is an aggregate of multiple annotators' judgments (see [Section 3 of our paper](https://arxiv.org/abs/2506.00314)).

**Turn Object (objects in `dialogue`)**

The `dialogue` list contains a sequence of turn items, each with the following structure:

*   **`turn_ind`**: The turn's chronological index (integer, starts at `0`).
*   **`role`**: The speaker, either `"USER"` or `"ASST"` (assistant).
*   **`utterance`**: The text content of the turn.
*   **`turn_level_aggregated`**: A dict of human annotations. Only assistant turns have this key. The same as `dial_level_aggregated`, each annotation is an aggregate of multiple annotators' judgments.

**Annotations**

The `turn_level_aggregated` structure provides turn-level scores based on the following aspects:
*   **`relevance`** (0-3): Does the assistant's response make sense and meet the user's interests?
*   **`interestingness`** (0-2): Does the response make the user want to continue the conversation?

The `dial_level_aggregated` structure provides dialogue-level scores based on the following aspects:
*   **`understanding`** (0-2): Does the assistant understand the user's request and try to fulfill it?
*   **`task_completion`** (0-2): Does the assistant make recommendations that the user finally accepts?
*   **`interest_arousal`** (0-2): Does the assistant try to spark the user's interest in something new?
*   **`efficiency`** (0-1): Does the assistant suggest items matching the user's interests within the first three interactions?
*   **`dialogue_overall`** (0-4): What is the overall impression of the assistant's performance?

## Example Snippet

The following snippet illustrates the structure of a single conversation object.

```python
[
  {
    # A unique identifier for the entire conversation.
    "conv_id": "barcor_redial_06c2c40a-921b-414a-85cc-2501469605cd",

    # An list containing the sequence of conversational turns.
    "dialogue": [
      {
        "turn_ind": 0,
        "role": "USER",
        "utterance": "Hello I need a movie that will make me laugh"
      },
      {
        "turn_ind": 1,
        "role": "ASST",
        "utterance": "Have you seen The Hangover (2009)?",

        # Turn-level annotations evaluating the assistant's specific response.
        "turn_level_aggregated": {
          "relevance": 2,
          "interestingness": 2
        }
      },
      # ... more turns
    ],

    # Dialogue-level annotations evaluating the assistant's overall performance.
    "dial_level_aggregated": {
      "understanding": 0,
      "task_completion": 0,
      "interest_arousal": 1,
      "efficiency": 1,
      "dialogue_overall": 1,
    }
  }
  # ... more conversations
]
```

## References
[1] WSDM '25: Bernard et al., [CRS Arena: Crowdsourced Benchmarking of Conversational Recommender Systems](https://dl.acm.org/doi/10.1145/3701551.3704120).