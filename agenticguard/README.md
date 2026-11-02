# AgenticGuard attack corpus

`attack-corpus-v2.json` is the corpus that ships in the AgenticGuard SDK, published here so you can read it and use it without the SDK. MIT-licensed, like the rest of this repository.

It measures something different from this benchmark. This benchmark asks whether an agent follows an injection. That corpus assumes the injection worked and asks what a policy layer lets through: each tool-call chain replays, in order, the calls a compromised model would make, with the decision expected at each one.

The format, the results and how to run it are in [the AgenticGuard docs](https://sentinelden.com/docs/agenticguard). The chains in this benchmark (`aib/data/attacks.jsonl`, family `tool_chain`) cover the same techniques in a form a model can run.
