We currently have two ways of modifying the LLM: adding noise and zeroing.

| % Modification    | % Noise    | Conclusion                                    |
|-------------------|------------|-----------------------------------------------|
| 100               | 10         | Nonsense long output for every layer          |
| 20               | 30         | Nonsense long output for every layer          |
| 15                | 10         | Nonsense short in many earlier, word-level error in later layers|
| 10               | 20         | Nonsense long output for every layer          |
| 10               | 15         | Nonsense long output for every layer          |
| 10                | 10         | Nonsense short in earlier, word-level error in later layers|
| 10                | 1         | Not enough modification |
| 1                | 10         | Not enough modification|

| % Modification    | Conclusion                                    |
|-------------------|-----------------------------------------------|
| 5               | Not enough modification          |
| 10               | Word changes          |
| 20               | Word changes         |

    
