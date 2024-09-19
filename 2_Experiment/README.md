We currently have two ways of modifying the LLM: adding noise and zeroing.

| % Modification    | % Noise    | Conclusion                                    |
|-------------------|------------|-----------------------------------------------|
| 100               | 10         | Nonsense long output for every layer          |
| 10                | 10         | Nonsense short in earlier, word-level error in later layers|
| 1                | 10         | Not enough modification |
| 10                | 11         | Not enough modification|


