# **Strategic Architecture for Global Market Selection: A Next-Generation MCDM and LLM-Assisted Ranking Pipeline** 

## **The Strategic Imperative of Market Selection** 

The evaluation, ranking, and sequencing of international markets for a leveraged derivatives and synthetic equity trading platform represents a highly dimensional, nonlinear optimization problem. Standard market-entry scorecards deployed in traditional corporate strategy rely heavily on static, linear weighting models and generalized macroeconomic indicators. These legacy methodologies systemically fail to capture the granular regulatory nuances, dynamic policy shifts, and strict capital constraints inherent to offshore virtual asset and derivatives platforms<sup>1</sup> . 

This report details an exhaustive, finalized methodological approach for a market ranking and sequencing model. The architecture is engineered to transcend subjective, heuristic-based scoring by integrating Large Language Model (LLM) regulatory extraction pipelines, advanced Multi-Criteria Decision Making (MCDM) algorithms, and stochastic simulation engines. The resulting framework fuses the "De Jure" automated regulatory extraction pipeline to mitigate LLM hallucinations<sup>3</sup> , the Bayesian Best-Worst Method (BWM) for probabilistic stakeholder weight elicitation<sup>4</sup> , the modified CRITIC-M method for objective data variance weighting<sup>6</sup> , and Stochastic Multicriteria Acceptability Analysis (SMAA) layered over a Robust TOPSIS algorithm to prevent mathematical rank reversal<sup>8</sup> . 

Crucially, the foundational logic of this entire architecture is dictated by the geographic domicile of the platform's founding entity. Because the originating operations are based in Bengaluru, India, the ranking pipeline must strategically incorporate the strict outbound capital controls enforced by the Reserve Bank of India (RBI) under the Foreign Exchange Management Act (FEMA) and the Liberalised Remittance Scheme (LRS)<sup>11</sup> . The report operationalizes the International Financial Services Centre (IFSC) at GIFT City as a critical structural gateway for outbound derivative exposure<sup>13</sup> , ensuring that the ranking model’s mathematical outputs translate directly into legally deployable strategies for an India-domiciled management team. 

## **Deconstructing the Bengaluru Constraint: Regulatory Realities and the GIFT City Catalyst** 

Mathematical elegance in market ranking is irrelevant if the resulting corporate strategy violates the sovereign laws governing the founding team. The ranking model's foundational 

parameters—specifically regarding capital deployment, leverage constraints, and crossborder payment rails—must be mapped against the stringent Indian macroeconomic controls that bind a Bengaluru-based enterprise. 

The Reserve Bank of India's Liberalised Remittance Scheme (LRS) permits resident Indians to remit up to USD 250,000 per financial year for permissible current or capital account transactions<sup>11</sup> . However, the LRS strictly and explicitly prohibits remittances intended for margin trading, leveraged products, foreign exchange trading, or speculative derivative instruments on overseas exchanges<sup>11</sup> . Furthermore, under the Prevention of Money Laundering Act (PMLA), the Financial Intelligence Unit (FIU-IND) has mandated that all offshore Virtual Asset Service Providers (VASPs) servicing Indian IP addresses must register as reporting entities, enforce physical-presence KYC protocols, and entirely cease the use of privacyenhancing technologies<sup>18</sup> . Failure to comply results in immediate DNS blocking, punitive fines, and potential criminal liability for domestic directors<sup>2</sup> . 

Consequently, a Bengaluru-based entity attempting to launch an offshore leveraged derivatives platform targeting resident Indian retail capital via direct fiat on-ramps (e.g., UPI) will immediately trigger FEMA violations and subsequent enforcement by the Enforcement Directorate (ED). A traditional market-entry model that ranks a jurisdiction like the UAE or Brazil as the optimal target based purely on their domestic crypto-friendliness<sup>1</sup> is strategically incomplete if it does not account for how capital legally moves from the Indian parent to the offshore subsidiary. 

To legally operationalize the launch in top-ranked offshore markets while maintaining a Bengaluru operational base, the corporate architecture must utilize the International Financial Services Centre (IFSC) at Gujarat International Finance Tec-City (GIFT City). Under the unified regulatory purview of the International Financial Services Centres Authority (IFSCA), a GIFT City unit is treated as a non-resident entity under FEMA, operating functionally as a foreign jurisdiction while remaining geographically within India<sup>20</sup> . Specifically, the establishment of a Category III Alternative Investment Fund (AIF) within GIFT IFSC serves as the optimal structural vehicle and fundamentally alters the parameters of the ranking model. Unlike Category I and II AIFs, which face strict limitations on borrowing, Category III AIFs are expressly permitted to undertake diverse and complex trading strategies, including the extensive use of leverage and investments in listed or unlisted derivatives<sup>14</sup> . By registering as a Fund Management Entity (FME) in GIFT City<sup>22</sup> , the Bengaluru team can legally pool compliant outbound investments from Indian institutions and High-Net-Worth Individuals under the FEMA Overseas Investment Rules (2022)<sup>24</sup> , or raise foreign capital directly. This structural realization actively shapes the ranking model's logic. The model no longer needs to heavily penalize jurisdictions that lack direct INR-to-fiat payment rails, as the capital formation occurs in USD within the GIFT City AIF<sup>14</sup> . The ranking model can therefore prioritize jurisdictions that offer the most favorable B2B licensing regimes for foreign funds, rather than focusing exclusively on direct-to-retail offshore loopholes that invite FIU-IND enforcement<sup>19</sup> . 

## **Defining the Decision Matrix: Finalized Indicators and Data Ingestion Strategy** 

A rigorous multi-criteria ranking model demands indicators that reflect product-specific realities. Relying on broad proxy variables introduces systemic bias; for instance, general cryptocurrency adoption indices (such as the Chainalysis Global Crypto Adoption Index) do not perfectly correlate with institutional demand for highly leveraged synthetic equity perpetuals<sup>1</sup> . The finalized indicator framework abandons pure macroeconomic proxies in favor of a specialized eight-criterion matrix (C1–C8) segmented into four domains: Commercial Potential, Regulatory Architecture, Operational Friction, and Sovereign Stability. 

### **The Finalized Indicator Matrix** 

The evaluation matrix comprises the following criteria, balanced between benefit indicators (where higher values increase the jurisdiction's rank) and cost indicators (where higher values decrease the jurisdiction's rank): 

|**Criterion**<br>**Code**|**Domain**|**Indicator**<br>**Name**|**Type**|**Measurem**<br>**ent**<br>**Construct**<br>**and**<br>**Justificatio**<br>**n**|**Target**<br>**Resolution**|
|---|---|---|---|---|---|
|**C1**|Commercia<br>l|Demand<br>Scale|Benefit|Replaces<br>generic<br>crypto-<br>adoption<br>indices with<br>derivative-<br>specific<br>trading<br>volume<br>proxies.<br>Utilizing the<br>Herfindahl-<br>Hirschman<br>Index (HHI)<br>for<br>exchange|Quantitativ<br>e<br>(Normalize<br>d scale 1-<br>10)|



||||concentrati<br>on helps<br>identify<br>markets<br>with<br>fragmented<br>,<br>contestable<br>derivative<br>volume<br>rather than<br>monopolies<br>1.||
|---|---|---|---|---|
|**C2**<br>Commercia<br>l|Revenue<br>Depth|Benefit|Evaluates<br>the capital<br>availability<br>per user,<br>measured<br>via Gross<br>National<br>Income<br>(GNI) per<br>capita<br>adjusted<br>for<br>purchasing<br>power,<br>paired with<br>World Bank<br>financial<br>inclusion<br>metrics<br>(e.g.,<br>mobile<br>money<br>account<br>penetration<br>)<sup>1</sup>.|Quantitativ<br>e<br>(Normalize<br>d scale 1-<br>10)|



|**C3**|Regulatory|Pathway<br>Clarity|Benefit|Measures<br>the<br>existence<br>of an<br>explicit, in-<br>force<br>regulatory<br>framework<br>specifically<br>tailored for<br>retail<br>leveraged<br>derivatives<br>and<br>synthetic<br>equities<br>(e.g., VARA<br>Part V in the<br>UAE, BCB<br>Resolutions<br>in Brazil)<sup>2</sup>.|Categorical<br>to Numeric<br>via Rule DB|
|---|---|---|---|---|---|
|**C4**|Regulatory|Leverage<br>Fit|Benefit|Quantifies<br>the<br>statutorily<br>permitted<br>retail<br>leverage<br>cap relative<br>to the<br>platform's<br>product<br>target. A 1:5<br>cap yields<br>low fit; a<br>1:300 cap<br>yields high<br>fit.|Quantitativ<br>e / Rule-<br>based|



|**C5**|Friction|Local Rail<br>Fit|Benefit|Evaluates<br>the<br>availability<br>and<br>openness<br>of instant<br>fiat<br>payment<br>rails to<br>virtual asset<br>service<br>providers,<br>considering<br>the<br>integration<br>potential<br>with the<br>GIFT City<br>USD-base.|Categorical<br>(High/Med/<br>Low)|
|---|---|---|---|---|---|
|**C6**|Friction|Capital<br>Burden|Cost|Represents<br>the<br>minimum<br>statutory<br>paid-up<br>capital and<br>licensing<br>fees<br>required for<br>a<br>derivatives/<br>VASP<br>license. This<br>operates as<br>the primary<br>knockout<br>threshold in<br>the<br>algorithmic|Quantitativ<br>e (USD<br>Equivalent)|



|||||pipeline<sup>2</sup>.||
|---|---|---|---|---|---|
|**C7**|Friction|Presence<br>Friction|Cost|Aggregates<br>mandates<br>for<br>localized<br>physical<br>infrastructu<br>re, local<br>resident<br>directors,<br>data<br>localization<br>laws, and<br>domestic<br>custody<br>constraints.|Categorical<br>to Numeric|
|**C8**|Sovereign|Regulatory<br>Volatility|Cost|Measures<br>the<br>frequency<br>of policy<br>reversals,<br>the<br>presence of<br>draft-stage<br>consultatio<br>ns versus<br>enacted<br>laws, and<br>subjective<br>predictabilit<br>y, stabilized<br>by World<br>Bank<br>governance<br>indices<sup>1</sup>.|Likert-scale<br>Expert<br>Judgment|



### **The Finalized Data Sourcing Architecture** 

To populate the C1–C8 matrix efficiently and repeatedly, the data ingestion architecture 

avoids the fragility of manual updates by relying on a trifecta of automated APIs, structured legal extractions, and robust secondary indices<sup>1</sup> . 

The foundation of the commercial and friction domains (C2, C5) relies on Macroeconomic & Development APIs. The World Bank Open Data API serves as the primary programmatic feed for financial inclusion, GNI, and mobile/internet penetration data<sup>1</sup> . This guarantees zero-cost, continuous, and highly structured data retrieval without the legal risks or maintenance overhead of web scraping. 

For the regulatory domains (C3, C4, C6, C7), Official regulatory publications form the absolute ground truth. Third-party scorecards fail to capture the specific leverage caps or synthetic equity classifications required for the decision matrix<sup>1</sup> . Sources include the Virtual Assets Regulatory Authority (VARA) rulebooks for the UAE, the Central Bank of Brazil (BCB) resolutions, OJK frameworks in Indonesia, and equivalent SEC publications in target jurisdictions<sup>2</sup> . Because these rules dictate the survival of the business model, they are ingested as raw, unstructured text directly from sovereign domains and processed via the LLM extraction pipeline. 

Finally, for Commercial Scale and Sovereign Stability (C1, C8), proprietary market data and institutional indices provide stabilization. Volumetric exchange data obtained via institutional data providers (such as Kaiko or CryptoCompare) combined with the Herfindahl-Hirschman Index (HHI) calculation for exchange concentration provide the basis for C1<sup>1</sup> . For regulatory volatility (C8), the World Bank's B-READY index and the Worldwide Governance Indicators (WGI) Regulatory Quality dimension serve as critical, objective cross-checks against subjective management estimates<sup>1</sup> . 

## **Automated Regulatory Intelligence: The LLM Extraction Pipeline** 

The most significant bottleneck in maintaining a dynamic decision matrix is the manual interpretation of dense, hierarchically structured foreign legal text<sup>3</sup> . Relying on zero-shot LLM prompts to read a regulatory document and directly assign a 1–10 score to a jurisdiction results in a documented accuracy ceiling of approximately 60% and is prone to severe hallucination<sup>1</sup> . Legal compliance requires absolute determinism, which raw generative models cannot guarantee. 

To resolve this, the finalized approach implements a scaled-down, production-ready adaptation of the "De Jure" pipeline, a fully automated, domain-agnostic framework for extracting structured regulatory rules<sup>3</sup> . The pipeline isolates fact-extraction from mathematical scoring, ensuring that the LLM only retrieves verifiable data parameters (e.g., leverage ratios, capital floors) rather than engaging in subjective scoring. 

### **Document Normalization and Retrieval Grounding** 

Raw PDFs and HTML pages scraped from international regulators are first pre-processed into structured Markdown<sup>3</sup> . This conversion preserves the hierarchical authority of the text— 

maintaining the structural integrity of Chapters, Sections, and Articles—which is critical for resolving legal cross-references<sup>30</sup> . The LLM is strictly constrained via Retrieval-Augmented Generation (RAG) to answer queries solely from the provided regulatory span, completely disabling reliance on parametric memory to prevent the hallucination of non-existent statutes<sup>1</sup> . 

### **Semantic Decomposition and JSON Schema Extraction** 

The LLM is instructed to parse the legal text into highly structured JSON rule units. Instead of prompting the model with an open-ended question such as "What is the capital requirement for a license?", the system enforces a strict schema requiring the model to extract highly specific key-value pairs: Entity_Type, Required_Capital_Local_Currency, Leverage_Cap_Retail, Asset_Class_Coverage, and crucially, the Verbatim_Source_Span<sup>3</sup> . 

### **Multi-Criteria LLM-as-a-Judge Evaluation** 

To combat judicial hallucination and extraction drift, a secondary "LLM-as-a-judge" mechanism evaluates the output generated during the extraction phase<sup>3</sup> . The judge model verifies the extracted JSON against 19 predefined criteria spanning metadata integrity, definition alignment, and rule semantics<sup>3</sup> . For example, the judge explicitly verifies whether the extracted leverage cap of "5x" actually exists within the verbatim source span provided by the extractor model. If the verbatim source span does not contain the extracted numeric fact, the JSON unit is flagged for rejection, preventing hallucinated thresholds from entering the decision matrix<sup>1</sup> . 

### **Confidence Tagging and Evidence Calibration** 

Extracted facts that successfully pass the LLM judge are fed into a deterministic rule database. To quantify the inherent uncertainty of legal interpretation for downstream stochastic simulations, the pipeline automatically assigns an evidence tag to each data point<sup>2</sup> : 

- **Verified Fact:** The fact passed the LLM judge without errors, and multiple independent extraction passes (utilizing multi-agent negotiation frameworks) generated identical JSON values, confirmed by a high Kendall's W coefficient of agreement<sup>1</sup> . The rule is confirmed as in-force law. 

- **Analyst Estimate:** The fact could not be deterministically extracted from primary legal text and relies on secondary sources, proxy metrics, or human analyst estimation. 

- **Unresolved Policy / Draft:** The LLM judge flagged a semantic contradiction, or the text explicitly denotes a draft or consultation phase (e.g., ongoing SEC proposals in Thailand)<sup>2</sup> . 

The raw extracted facts (such as absolute leverage limits) are then deterministically mapped to the 1–10 matrix scores using fixed, human-defined mathematical thresholds, completely insulating the final numeric matrix from direct generative AI influence. 

## **Eliciting and Synthesizing Criteria Weights: Bayesian** 

## **BWM and CRITIC-M** 

In traditional Multi-Criteria Decision Making, assigning relative weights to the chosen criteria is highly vulnerable to subjective bias. The standard assumption of equal weighting is 

mathematically negligent, while arbitrary percentage allocations fail under scrutiny. To create a finalized, mathematically rigorous weighting schema, the architecture synthesizes subjective stakeholder priorities using the Bayesian Best-Worst Method (BBWM) with objective, datadriven weights derived from the Modified CRITIC method (CRITIC-M). 

### **Subjective Elicitation: The Bayesian Best-Worst Method (BBWM)** 

The traditional Analytic Hierarchy Process (AHP) requires pairwise comparisons. For an eight-criterion matrix, this demands 28 separate judgments, leading rapidly to cognitive fatigue, contradictory logic, and violations of transitivity among executive decision-makers<sup>33</sup> . The Best-Worst Method (BWM) streamlines this process by requiring decision-makers to identify only the absolute Best (most important) and Worst (least important) criteria overall. The executives then compare all other criteria relative only to these two references, reducing 

the requirement to comparisons (just 13 judgments for eight criteria)<sup>4</sup> . However, traditional linear BWM yields a single point estimate and struggles to smoothly aggregate conflicting opinions among multiple founders or stakeholders. The finalized pipeline therefore adopts the Bayesian BWM<sup>4</sup> . 

In BBWM, the integer-based pairwise comparisons provided by the decision-making group are modeled using a multinomial likelihood distribution<sup>37</sup> . The final aggregated weights of the criteria are represented as probability distributions rather than static scalar values. A Dirichlet distribution is selected as the prior over the weight vectors because it natively satisfies the fundamental MCDM constraint that all criteria weights must be non-negative and sum to exactly 1<sup>4</sup> . 

The joint probability distribution for group decision-making is evaluated using Markov Chain Monte Carlo (MCMC) sampling techniques, specifically employing Just Another Gibbs Sampler (JAGS)<sup>5</sup> . The MCMC algorithm generates tens of thousands of posterior draws, allowing the system to verify chain convergence via Gelman-Rubin diagnostics<sup>41</sup> . This yields a central weight vector alongside credible intervals that quantify the exact degree of consensus or dissent among the management team regarding the relative importance of factors like Regulatory Volatility versus Demand Scale<sup>42</sup> . 

### **Objective Data-Driven Weighting: CRITIC-M** 

To balance the subjective priors extracted from the executive team, the model integrates the Criteria Importance Through Intercriteria Correlation (CRITIC) method, specifically its advanced modification, CRITIC-M<sup>6</sup> . CRITIC derives objective weights by analyzing the statistical structure of the decision matrix itself—specifically the contrast intensity (measured by standard deviation) and the conflicting relationships (measured by correlation)<sup>43</sup> . The standard CRITIC method utilizes linear min-max normalization, which forces the bounds of every criterion to exactly 0 and 1. This significantly amplifies the impact of extreme outliers and artificially inflates the standard deviation, warping the resulting weights<sup>7</sup> . CRITIC-M resolves this by employing an enhanced normalization procedure that minimizes artificial deviations<sup>6</sup> . 

The objective weight W; for criterion j is calculated by first determining the amount of information Cj contained in the criterion: 



<!-- Start of picture text -->
Ci = 0; Sa mij)<br>i=l<br><!-- End of picture text -->

where is the standard deviation of the normalized criterion , and is the correlation tj coefficient between criteria and . The final objective weight is simply the normalized information content: 



<!-- Start of picture text -->
W= aeC.<br>ja Cj<br><!-- End of picture text -->

Criteria that exhibit high variance (indicating strong discriminatory power between jurisdictions) and low correlation with other criteria (indicating they provide unique, nonredundant information) are awarded higher objective weights<sup>7</sup> . 

### **The Unified Weighting Vector** 

The final working weight vector used in the primary ranking algorithm is a convex weBWM combination of the posterior mean weights derived from BBWM ( ) and the objective weRITIC weights from CRITIC-M ( ): 



<!-- Start of picture text -->
w} = awhawa +(1- a)woRiTIC<br><!-- End of picture text -->

where a€ (0,1) is a tunable hyperparameter defining the corporate reliance on expert intuition versus empirical matrix variance. This fusion ensures that the model respects the strategic intent of the founders while remaining grounded in the mathematical reality of the data. 

## **The Ranking Engine: Robust TOPSIS and the Eradication of Rank Reversal** 

The Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) serves as the primary computational ranking engine. TOPSIS evaluates alternative jurisdictions based on their geometric distance to a Positive Ideal Solution (PIS) and a Negative Ideal Solution (NIS)<sup>2</sup> . 

### **The Algorithmic Execution** 

Before mathematical scoring begins, the pipeline executes a hard knockout screen. Any jurisdiction where the Capital Burden (C6) exceeds the liquid runway constraints of the startup is immediately purged from the matrix. For example, Vietnam's requirement for approximately USD 380 million in charter capital constitutes a hard constraint, not merely a low score<sup>2</sup> . 

Following the knockout, the remaining decision matrix is normalized. Vector normalization is strictly utilized over sum or min-max normalization to mitigate severe scale distortions between criteria measured in different units: 



The weighted normalized decision matrix is generated by multiplying the normalized scores : by the unified weight vector 



The PIS ( ) and NIS ( ) are identified across all criteria, ensuring that benefit criteria seek maximums while cost criteria seek minimums. The Euclidean distances from the ideal ( ) and anti-ideal ( ) are computed for each jurisdiction, yielding the final closeness coefficient : 



where . The market with the highest closeness coefficient is ranked first<sup>2</sup> . **Resolving Rank Reversal via TOPSIS-RAD** 

A critical systemic vulnerability in standard TOPSIS, long debated in operations research literature, is the phenomenon of "rank reversal"<sup>33</sup> . Rank reversal occurs when the relative ranking of Market A and Market B flips simply because an irrelevant Market C is added to or removed from the dataset<sup>33</sup> . This mathematical instability occurs because the PIS and NIS in standard TOPSIS are dynamically derived from the highest and lowest scores observed in the current dataset; altering the pool of countries physically alters the geometric coordinates of the ideal frontiers<sup>9</sup> . 

To ensure absolute stability, the pipeline integrates principles from TOPSIS-RAD (Reference 

points and Acceptability Dimensions)<sup>9</sup> . Instead of allowing the dataset to define the extremes, the model anchors the geometric space utilizing Vetoed Performance Levels (VPL) and Desired Performance Levels (DPL) established independently by the management team. By fixing the ideal and anti-ideal vectors as absolute, domain-defined constants (e.g., an absolute regulatory certainty score is permanently fixed at 10; a total ban is permanently fixed at 1) rather than dynamic relative maximums, the geometric space is anchored<sup>9</sup> . Consequently, adding a newly researched market (e.g., Turkey or Kenya) to the pipeline will not distort the geometric distances of previously evaluated markets like the UAE or Brazil, permanently resolving the rank reversal vulnerability<sup>2</sup> . 

## **Strategic Optimization: The Capital-Constrained Knapsack Formulation** 

While the TOPSIS engine effectively generates a sequential ranking of markets, simply moving down an ordinal list (e.g., expanding to Market #1, then Market #2, then Market #3) is strategically flawed for a startup operating under severe capital constraints<sup>1</sup> . Two highly efficient, mid-ranked markets might collectively generate superior revenue and consume less capital than a single, top-ranked but capital-intensive market<sup>49</sup> . 

To solve this, the pipeline treats market sequencing as a Multiple-Choice Multi-Period Knapsack Problem (MCMPKP)<sup>50</sup> . In this formulation, the "knapsack" represents the total capital and legal-hour budget available for a specific expansion wave. Each market represents an item with a specific "weight" (the capital burden from criterion C6 plus operational launch costs) and a specific "value" (derived from the TOPSIS closeness coefficient combined with expected 12-month contribution margins)<sup>2</sup> . 

The algorithm maximizes the total value of the selected markets subject to the constraint that the combined capital weight cannot exceed the budget for that wave<sup>50</sup> . Furthermore, multiple-choice constraints are applied to prevent mutually exclusive strategic errors (e.g., dedicating capital to two competing Southeast Asian hubs simultaneously when only one regional HQ is required). This transforms the output from a static ranking into a dynamic, optimized portfolio of markets clustered into sequential expansion waves. 

## **Stochastic Simulation and Decision Robustness (Module C)** 

Single-point estimates in MCDM create a dangerous illusion of precision. To ensure that strategic entry decisions are resilient against both human estimation errors and macroeconomic unpredictability, the pipeline deploys a sophisticated stochastic robustness engine<sup>1</sup> . 

### **Stochastic Multicriteria Acceptability Analysis (SMAA)** 

SMAA explores the entire feasible weight space to determine the stability of the rankings under complete or partial weight uncertainty<sup>10</sup> . By utilizing the open-source Python library 

PySensMCDA (v2.0), the pipeline efficiently performs SMAA-TOPSIS without the need for bespoke algorithmic development<sup>8</sup> . 

The algorithm leverages advanced Hit-And-Run Monte Carlo sampling over the bounded weight simplex<sup>8</sup> . Instead of outputting a single, brittle ranking, it generates a comprehensive Rank Acceptability Matrix, calculating the exact probability that a specific market achieves rank 



across all valid, randomly generated permutations of criteria importance<sup>10</sup> . 

Furthermore, SMAA calculates the Central Weight Vector ( ), representing the expected center of gravity of the favorable weight space that makes a specific alternative the winner<sup>10</sup> . If the model determines that Brazil ranks first, the central weight vector explicitly reveals the specific operational philosophy (e.g., a heavy bias toward Demand Scale over Regulatory Clarity) that mathematically forces Brazil into the top spot<sup>10</sup> . The executive team can immediately verify if these implied priorities align with actual corporate risk tolerance. 

### **Evidence-Calibrated Monte Carlo** 

Traditional Monte Carlo models applied to MCDM matrices typically apply uniform noise (e.g., 

perturbing all scores by points) across the entire matrix<sup>2</sup> . This approach is mathematically unsound; a legally verified minimum capital requirement extracted directly from a central bank's gazette does not possess the same uncertainty as a subjective estimate of local payment rail readiness<sup>1</sup> . 

The finalized pipeline deploys an Evidence-Calibrated Noise Model. The uncertainty parameter applied to each score during the Monte Carlo simulation is directly governed by the evidence tag generated during the LLM extraction phase<sup>1</sup> : 

- **Verified Fact:** Minimal variance is applied, utilizing a tight Gaussian distribution. 

- **Analyst Estimate:** Moderate variance is applied, utilizing a wider Gaussian distribution to account for proxy inaccuracies. 

- **Unresolved Policy / Draft:** Modeled using a bimodal distribution. For instance, if it is unclear whether synthetic equities fall under a lenient crypto regulator or a strict securities regulator (a known issue in several target markets), the score does not jitter uniformly. Instead, it flips probabilistically between the two specific, discrete outcome states<sup>2</sup> . 

This calibration guarantees that the Monte Carlo output accurately reflects true systemic and legal risk rather than artificial mathematical noise, providing a highly defensible confidence interval for the final ranking. 

## **Synthesis and Final Implementation Architecture** 

The finalized, end-to-end framework operates in a continuous, automated loop, seamlessly integrating automated legal extraction, mathematical optimization, and strategic sovereign compliance: 

The architecture begins with the ingestion of data streams. APIs continuously retrieve 

macroeconomic indicators, while targeted scrapers retrieve raw regulatory publications from target central banks. The De Jure LLM pipeline normalizes these texts, enforces JSON schema compliance, utilizes an LLM-as-a-judge for hallucination filtering, and outputs verifiable numeric values and evidence tags into the rule database<sup>1</sup> . 

Simultaneously, the management team executes a streamlined BWM survey. MCMC sampling fits a Dirichlet posterior, outputting probabilistic subjective weights, while the CRITIC-M algorithm parses the updated matrix to calculate objective variance weights<sup>4</sup> . These are mathematically coupled to form the foundational weight vector<sup>55</sup> . 

The robust ranking engine takes over, executing the capital knockout algorithm to purge unviable jurisdictions. The vector-normalized TOPSIS-RAD algorithm computes geometric distances against fixed, predefined veto boundaries, completely insulating the system from rank reversal<sup>9</sup> . The output is then fed into the multiple-choice knapsack optimizer to construct capital-efficient expansion waves<sup>49</sup> . 

Finally, the entire model is subjected to severe stochastic stress-testing. SMAA draws millions of random weight distributions to build the Rank Acceptability Matrix, while the EvidenceCalibrated Monte Carlo applies tag-dependent variance to validate the statistical significance of the top-ranked jurisdiction<sup>2</sup> . 

The resulting output is not merely an ordered list, but a legally viable, mathematically proven expansion roadmap. By recognizing the constraints of Indian capital controls and structuring the expansion through a Category III AIF at GIFT City<sup>14</sup> , the architecture provides a definitive, operational blueprint that allows a Bengaluru-based enterprise to legally dominate offshore leveraged derivatives markets. 

#### **Works cited** 

1. key points.txt 

2. mochatrade_ranking_model_context.md 

3. De Jure: Iterative LLM Self-Refinement for Structured Extraction of, <u>https://arxiv.org/html/2604.02276v1</u> 

4. Bayesian best-worst method - TU Delft Research Portal, <u>https://research.tudelf.nl/ft</u> iles/72980603/1_s2.0_S0305048318313963_main.pdf 

5. Modeling the supply chain sustainability imperatives in the fashion, <u>https://pmc.ncbi.nlm.nih.gov/articles/PMC11687714/</u> 

6. modification of the critic method using fuzzy rough numbers, <u>https://scidar.kg.ac.rs/bitstream/123456789/15709/1/2022%20-DMAME.pdf</u> 

7. (PDF) Objective methods for determining criteria weight coefficients, <u>https://www.researchgate.net/publication/346168287_Objective_methods_for_d etermining_criteria_weight_coefcients_A_modiffi cation_of_the_CRITIC_methodi</u> 

8. New tool PySensMCDA enables sensitivity analysis in multi-criteria, <u>https://bioengineer.org/new-tool-pysensmcda-enables-sensitivity-analysis-inmulti-criteria-decision-problems/</u> 

9. TOPSIS-RAD: RANKING ACCORDING TO DESIRES - arXiv, <u>https://arxiv.org/pdf/2606.07253</u> 

10. (PDF) Implementing Stochastic Multicriteria Acceptability Analysis, <u>https://www.researchgate.net/publication/222672268_Implementing_Stochastic_ Multicriteria_Acceptability_Analysis</u> 

11. Liberalised Remittance Scheme (LRS): Meaning, Rules, Limits and, <u>https://groww.in/blog/liberalised-remitancet</u> -scheme 

12. Notifications| Official Website of Reserve Bank of India, <u>https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=8100&Mode=0</u> 

13. Investment Funds 2026 - Chambers Practice Guides, <u>https://practiceguides.chambers.com/practice-guides/comparison/1112/18207/28 479-28480-28481-28482</u> 

14. GIFT City AIFs: 2026 Tax Benefits & Setup Guide - Enterslice, <u>https://enterslice.com/learning/gift-city-aifs-setup-guide/</u> 

15. RBI/FED/2015-16/3 FED Master Direction No. 7 - LexComply, <u>https://lexcomply.com/rsjadmin/news/201705102317Master%20Direction%20%20Liberalised%20Remittance%20Scheme.PDF</u> 

16. An Overview Of LRS - Legal 500, <u>https://www.legal500.com/en/intelligence/india/fnancei</u> -and-banking/an- 

   - <u>overview-of-lrs</u> 

17. LRS for sending money abroad from India - ICICI Bank, <u>https://www.icici.bank.in/nri-banking/nriedge/nri-articles/lrs-for-sending-moneyabroad-from-india</u> 

18. Web3 Regulatory Intelligence — Global Dashboard, https://complianex- <u>regintel.pplx.app/</u> 

19. India's FIU-IND puts crypto under full AML scope with strict KYC rules, <u>https://www.bitget.com/news/detail/12560605147775</u> 

20. Doing Business in Gujarat - GIFT City, https://api.giftgujarat.in/public/tool- <u>guiedes-for-seting/DoingBusinessInGIFTCity.pdft</u> 

21. Gift City - PMS Bazaar, https://pmsbazaar.com/gift-funds 

22. New Fund Management Regulations Issued For IFSC-GIFT City To, <u>https://elplaw.in/leadership/new-fund-management-regulations-issued-for-ifscgift-city-to-overhaul-fund-governance/</u> 

23. GIFT City AIF & FME Lawyers | Fund Structuring IFSC | Candour Legal, <u>https://candourlegal.com/gift-city-aif-fme-lawyers/</u> 

24. International wealth management solutions - LGT Wealth India, <u>https://www.lgtindia.in/en/wealth-management/international-business</u> 

25. Navigating FEMA, LRS and ODI for Startup Investments - CAclubindia, <u>https://www.caclubindia.com/articles/globalising-your-portfolio-navigatingfema-lrs-and-odi-for-startup-investments-54945.asp</u> 

26. Family Offices in India – The Complete Guide - Treelife, <u>https://treelife.in/legal/family-ofcesfi</u> -in-india/ 

27. GIFT City AIF Fund | Monarch India Growth Fund IFSC, 

<u>https://www.mnclgroup.com/aif/gift-city</u> 

28. CoinGlass 2025 Crypto Derivatives Market Annual Report, <u>https://www.coinglass.com/learn/2025-annual-report-en</u> 

29. Investigating and Understanding the Unreliability of LLM-as-a-Judge, <u>https://www.researchgate.net/publication/397509649_Judicial_Hallucination_Inv estigating_and_Understanding_the_Unreliability_of_LLM-as-a-Judge</u> 

30. Citation-Closure Retrieval and Per-Rule Attribution for Real ... - arXiv, <u>https://arxiv.org/html/2605.29742v1</u> 

31. Best hallucination detection tools for LLM applications (2026), <u>https://www.braintrust.dev/articles/best-hallucination-detection-tools-2026</u> 

32. De Jure: Iterative LLM Self-Refinement for Structured Extraction of, <u>https://arxiv.org/abs/2604.02276</u> 

33. The rank reversal problem in multi-criteria decision making, <u>https://www.researchgate.net/publication/326980662_THE_RANK_REVERSAL_PR OBLEM_IN_MULTI-CRITERIA_DECISION_MAKING_A_LITERATURE_REVIEW</u> 

34. Analytical Hierarchy Process (AHP) A Multi-Criteria Decision, <u>https://eacademic.ju.edu.jo/E.abutaieh/Lists/Published%20Research/ Atachments/34/p09_ahp.pdft</u> 

35. Best-worst multi-criteria decision-making method - ResearchGate, <u>https://www.researchgate.net/publication/287375148_Best-worst_multicriteria_decision-making_method_Some_properties_and_a_linear_model</u> 

36. A Bayesian BWM and VIKOR-based model for assessing hospital, <u>https://pmc.ncbi.nlm.nih.gov/articles/PMC8593641/</u> 

37. (PDF) Bayesian Best-Worst Method: A Probabilistic Group Decision, <u>https://www.researchgate.net/publication/333694447_Bayesian_BestWorst_Method_A_Probabilistic_Group_Decision_Making_Model</u> 

38. Exploring the Role of Multiple Criteria Decision-Making in Enterprise, <u>https://ijb.cyut.edu.tw/var/fle/10/1010/img/1013/V28N306IJBi</u> -2023-03-05.pdf 

39. Evaluation of Potential Logistics Village Alternatives Using Bayesian, <u>https://opt.reapress.com/journal/article/download/35/255/501</u> 

40. Bridge infrastructure resilience assessment against seismic hazard, <u>https://cdnsciencepub.com/doi/10.1139/cjce-2021-0503</u> 

41. Investigating key performance indicators for emergency healthcare, <u>https://www.tandfonline.com/doi/full/10.1080/15623599.2026.2648079</u> 

42. Uneven Flows: Bayesian Best–Worst Method and GIS Analysis of, <u>https://www.mdpi.com/2073-445X/15/8/1357</u> 

43. CRITIC Method for MCDM Weighting | PDF - Scribd, <u>https://www.scribd.com/document/479065163/CRITIC</u> 

44. CRITIC Method | KÜRE Encyclopedia, https://kureansiklopedi.com/en/detay/critic- <u>method-5ebb2</u> 

45. decideXpert: Collaborative system using AHP-TOPSIS and fuzzy, <u>https://www.researchgate.net/publication/387495674_decideXpert_Collaborativ e_system_using_AHP-</u> 

<u>TOPSIS_and_fuzzy_techniques_for_multicriteria_group_decision-making</u> 

46. Early detection of students' failure using Machine Learning techniques, <u>https://www.econstor.eu/bitstream/10419/325777/1/S2214716023000271.pdf</u> 

47. (PDF) Resolving Rank Reversal in TOPSIS: A Comprehensive, <u>https://www.researchgate.net/publication/385848253_Resolving_Rank_Reversal_i n_TOPSIS_A_Comprehensive_Analysis_of_Distance_Metrics_and_Normalization_ Methods</u> 

48. Full article: The state of the art development of AHP (1979–2017), <u>https://www.tandfonline.com/doi/full/10.1080/00207543.2017.1334976</u> 

49. The use of the multiple knapsack problem in strategic management, <u>https://www.emerald.com/ijem/article/33/2/335/118820/The-use-of-the-multipleknapsack-problem-in</u> 

50. The Multiple-Choice Knapsack Problem | Operations Research, <u>https://pubsonline.informs.org/doi/10.1287/opre.27.3.503</u> 

51. The multiple-choice multi-period knapsack problem - ResearchGate, <u>https://www.researchgate.net/publication/245280590_The_multiplechoice_multi-period_knapsack_problem</u> 

52. Optimal Advertisement Placement Slot Using Knapsack Problem (A, <u>https://www.ijera.com/papers/Vol7_issue4/Part-4/F0704044662.pdf</u> 

53. SMAA-2: Stochastic Multicriteria Acceptability Analysis for Group, <u>https://www.researchgate.net/publication/31594359_SMAA-2_Stochastic_Multicr iteria_Acceptability_Analysis_for_Group_Decision_Making</u> 

54. pymcdm—The universal library for solving multi-criteria decision, <u>https://www.researchgate.net/publication/369475022_pymcdmThe_universal_library_for_solving_multi-criteria_decision-making_problems</u> 

55. Integrating Main Path Analysis and Hybrid MCDM to Evaluate, <u>https://www.mdpi.com/2076-3417/16/17/8545</u> 

