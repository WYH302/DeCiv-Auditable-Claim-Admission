# Source attribution and scope

The files provided here reproduce the shared-input experimental evaluation. They
contain the categorical validator votes, reference labels, governance metadata flags
and source/parent identifiers across the 1,800 evaluation packets. example_packets.csv
contains the two worked FEVER-derived packets illustrated in Table I, retaining
sentence locators and source URLs.

Upstream benchmark sources:

- FEVER: Thorne, Vlachos, Christodoulopoulos and Mittal (2018),
  https://aclanthology.org/N18-1074/ . Official annotation terms incorporate
  Wikipedia article-specific licenses, with CC BY-SA 3.0 where unavailable:
  https://fever.ai/download/fever/license.html .
- TruthfulQA: Lin, Hilton and Evans (2022),
  https://aclanthology.org/2022.acl-long.229/ . Official repository license:
  https://github.com/sylinrl/TruthfulQA/blob/main/LICENSE (Apache 2.0).
- HotpotQA: Yang et al. (2018), https://aclanthology.org/D18-1259/ .
  Official source and license: https://hotpotqa.github.io/ (CC BY-SA 4.0).
- Wikipedia revision identifiers: https://en.wikipedia.org/wiki/Wikipedia:Copyrights .

The evaluation corpus provides four structured packet variants per parent claim,
governance metadata flags, semantic and policy-reference action definitions, and
multi-validator votes. The code in vendor/deciv2/r12 implements the consensus
aggregators and risk routers evaluated in the manuscript, including the classical
Dawid and Skene (1979) expectation-maximization algorithm.
