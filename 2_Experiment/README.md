We currently have two ways of modifying the LLM: adding noise and zeroing.

Adding Noise

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

Zeroing

| % Modification    | Conclusion                                    |
|-------------------|-----------------------------------------------|
| 5               | Not enough modification          |
| 10               | minor word changes          |
| 20               | minor word changes         |
| 30               | minor word changes         |
| 40               | minor word changes         |
| 50               | minor word changes         |
| 60               | minor word changes         |



    
