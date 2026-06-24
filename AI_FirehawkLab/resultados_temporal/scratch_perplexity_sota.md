Here is an annotated set of 18 key papers (plus 2 foundational reviews) on wildfire dispatch prediction, resource allocation, and decision-support systems, focused on 2022–2025 and closely aligned with your four interests (tiering, two‑stage/imbalanced dispatch, temporal validation, concurrent-fire features). Because recent journal articles are copyrighted, I cannot reproduce verbatim quotes; instead I provide short, decision-support–focused paraphrases of the authors’ own framing. [cdnsciencepub](https://cdnsciencepub.com/doi/10.1139/er-2020-0019)

***

## Wildland aerial dispatch and initial-attack success

**1. Taylor, S. W., & Nadeem, K. (2022). “Predicting daily initial attack aircraft targets in British Columbia.” *International Journal of Wildland Fire*, 31(4), 449–468. DOI: 10.1071/WF21090.** [ouci.dntb.gov](https://ouci.dntb.gov.ua/en/works/lRjZPejl/)

- **Scope & methods.** Uses a grid-cell × day lasso–logistic modeling framework to predict where aircraft will be used on initial attack, combining daily fire occurrence models with conditional models of airtanker and helicopter use. [ouci.dntb.gov](https://ouci.dntb.gov.ua/en/works/lRjZPejl/)
- **Decision-support relevance (paraphrased).** The authors argue their models can be used in preparedness planning to anticipate “aircraft IA targets” one or more days ahead, supporting pre-positioning and staffing decisions in the provincial aviation program. [ouci.dntb.gov](https://ouci.dntb.gov.ua/en/works/lRjZPejl/)

***

**2. Wheatley, M., Wotton, B. M., Woolford, D. G., & Martell, D. L. (2023). “Modelling decisions concerning the dispatch of airtankers for initial attack on forest fires in Ontario, Canada.” *Canadian Journal of Forest Research*, 53(4), 217–233. DOI: 10.1139/cjfr-2022-0225.** [discover.research.utoronto](https://discover.research.utoronto.ca/26268-david-martell/publications)

- **Scope & methods.** Analyzes historical airtanker dispatch decisions in Ontario using statistical/machine-learning models to estimate the probability of dispatching 0, 1, 2, or more airtankers as a function of fire characteristics, location, and environmental conditions. [ouci.dntb.gov](https://ouci.dntb.gov.ua/en/works/lDXGgezl/)
- **Decision-support relevance (paraphrased).** They highlight that dispatch models can act as advisory tools to check for under‑ or over‑dispatch relative to historical practice and to expose implicit “risk-averse” tendencies when managers send more aircraft than the model predicts. [sciencedirect](https://www.sciencedirect.com/org/science/article/pii/S0045506722000797)
  - Imbalance is explicitly discussed: models perform well for none/one/two airtankers but underpredict larger fleets, underscoring the need for class-aware or two-stage strategies for rare, high‑resource events. [ouci.dntb.gov](https://ouci.dntb.gov.ua/en/works/lDXGgezl/)

***

**3. Cardil, A., Jiménez-Ruano, A., Monedero, S., et al. (2025). “Assessing the suppression difficulty of wildland fires for initial attack response.” *International Journal of Wildland Fire*, 34, WF24160. DOI: 10.1071/WF24160.** [nwfirescience](https://www.nwfirescience.org/biblio/assessing-suppression-difficulty-wildland-fires-initial-attack-response)

- **Scope & methods.** Introduces the Initial Attack Assessment (IAA) index (1–5) derived from 26,907 California ignitions via automated fire simulations; GLMs link IAA, a Fire Behavior Index, Terrain Difficulty Index, and response time to initial-attack success probabilities. [nwfirescience](https://www.nwfirescience.org/sites/default/files/publications/wf24160.pdf)
- **Decision-support relevance (paraphrased).** The paper positions IAA as a simple, user-facing index designed to plug directly into decision-support systems, allowing dispatchers to prioritize fires likely to exceed suppression capacity and to escalate resources when IAA is high despite short response times. [nwfirescience](https://www.nwfirescience.org/sites/default/files/publications/wf24160.pdf)
  - The authors note that IAA “feeds” existing DSS platforms and utility wildfire-risk tools, explicitly framing it as an operational triage layer rather than just an analytic metric. [nwfirescience](https://www.nwfirescience.org/biblio/assessing-suppression-difficulty-wildland-fires-initial-attack-response)

***

**4. Rodrigues, M., Alcasena, F. J., & Vega-García, C. (2019). “Modeling initial attack success of wildfire suppression in Catalonia, Spain.” *Science of the Total Environment*, 666, 915–927. DOI: 10.1016/j.scitotenv.2019.02.323.** [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/30818214/)

- **Scope & methods.** Random-forest models predict probability of containment given georeferenced ignitions, access, resources, and weather; “fire simultaneity” (number of concurrent fires) is an explicit predictor, along with wind, temperature and distance to stations. [zaguan.unizar](https://zaguan.unizar.es/record/112180/files/texto_completo.pdf)
- **Decision-support relevance (paraphrased).** The authors emphasize that their IA-containment probability maps can inform pre‑positioning of brigades and help identify days and areas where simultaneous fires plus extreme weather make escapes likely, thus guiding proactive surge capacity. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/30818214/)
  - This is one of the clearest demonstrations of including concurrent‑fires signals as engineered features directly tied to dispatch and pre‑positioning.  

***

**5. Alawode, G. L., Gelabert, P. J., & Rodrigues, M. (2025). “A spatially explicit containment modelling approach for escaped wildfires in a Mediterranean climate using machine learning.” *Geomatics, Natural Hazards and Risk*, 16(1), 2447514. DOI: 10.1080/19475705.2024.2447514.** [zaguan.unizar](https://zaguan.unizar.es/record/148652/files/texto_completo.pdf)

- **Scope & methods.** Uses random forests on 124 historical perimeters and geospatial data to estimate containment probability at 30‑m resolution for escaped fires, identifying where suppression is most likely to succeed or fail under given topography and fuels. [zaguan.unizar](https://zaguan.unizar.es/record/148652/files/texto_completo.pdf)
- **Decision-support relevance (paraphrased).** The authors frame their maps as tools for incident commanders to prioritize where to anchor control lines and concentrate crews, arguing that spatial patterns of containment probability can be integrated into operational planning and risk‑informed deployment. [zaguan.unizar](https://zaguan.unizar.es/record/148652/files/texto_completo.pdf)

***

**6. Taylor, S. W., et al. (2022). “Predicting daily initial attack aircraft targets in British Columbia.” *International Journal of Wildland Fire*, 31(4), 449–468. DOI: 10.1071/WF21090.** [ouci.dntb.gov](https://ouci.dntb.gov.ua/en/works/lRjZPejl/)

*(Included above as #1; kept here thematically because it explicitly links daily probabilistic aircraft targets to preparedness-level resource allocation.)*  

- **Two‑stage structure (paraphrased).** The work effectively uses a two‑stage pipeline—fire-occurrence prediction followed by conditional aircraft-use models—to obtain the marginal distribution of daily aircraft “targets”, illustrating a clean separation between ignition modelling and resource-demand modelling under rarity/imbalance. [ouci.dntb.gov](https://ouci.dntb.gov.ua/en/works/lRjZPejl/)

***

## Optimization-based dispatch and resource allocation

**7. Mendes, A. B., & Alvelos, F. (2023). “Iterated local search for the placement of wildland fire suppression resources.” *European Journal of Operational Research*, 304(3), 887–900. DOI: 10.1016/j.ejor.2022.04.037.** [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0377221722003502)

- **Scope & methods.** Formulates a mixed-integer programming model for locating suppression resources on a gridded landscape and solves it with an iterated local search metaheuristic to maximize coverage and minimize expected damage across candidate ignition scenarios. [ideas.repec](https://ideas.repec.org/a/eee/ejores/v304y2023i3p887-900.html)
- **Decision-support relevance (paraphrased).** The paper is explicit that the model is intended as a planning‑level decision-support tool for agencies to test alternative base configurations and quantify the marginal benefit of adding or relocating resources under limited budgets. [repositorium.uminho](https://repositorium.uminho.pt/entities/publication/2575b4f5-2be4-4fa4-880c-1bb3f60c1f29)

***

**8. Matos, M. A., Rocha, A. M. A. C., Costa, L., & Alvelos, F. (2023). “Resource Dispatch Optimization for Firefighting Based on Genetic Algorithm.” In *Computational Science and Its Applications – ICCSA 2023 Workshops*, LNCS 13940, 357–371. DOI: 10.1007/978-3-031-37108-0_28.** [repositorium.uminho](https://repositorium.uminho.pt/bitstreams/593d9440-df95-4fa3-b211-aab0397ae18e/download)

- **Scope & methods.** Models dispatch of 7 firefighting resources to 20 ignitions and uses a genetic algorithm to minimize total burned area, exploring different crossover, mutation, and selection operators tailored to a combinatorial dispatch schedule. [repositorium.uminho](https://repositorium.uminho.pt/bitstreams/593d9440-df95-4fa3-b211-aab0397ae18e/download)
- **Decision-support relevance (paraphrased).** The authors describe the model as a way to experiment with dispatch strategies under multiple simultaneous ignitions, stressing that such optimization can reveal more efficient deployment patterns than manual heuristic plans under complex, concurrent scenarios. [semanticscholar](https://www.semanticscholar.org/paper/Resource-Dispatch-Optimization-for-Firefighting-on-Matos-Rocha/00ce4f4a31c2e34dd96a89cae987e08b6892cad5)

***

**9. Suarez, D., Gomez, C., Medaglia, A. L., Akhavan-Tabatabaei, R., & Grajales, S. (2024). “Integrated Decision Support for Disaster Risk Management: Aiding Preparedness and Response Decisions in Wildfire Management.” *Information Systems Research*, 35(2), 609–628. DOI: 10.1287/isre.2022.0118.** [pubsonline.informs](https://pubsonline.informs.org/doi/10.1287/isre.2022.0118)

- **Scope & methods.** Proposes an analytics-centered framework that couples probabilistic risk assessment with optimization-based simulation of response, and applies it to wildfire preparedness and response decisions in Uruguay. [research.sabanciuniv](https://research.sabanciuniv.edu/id/eprint/49782/)
- **Decision-support relevance (paraphrased).** The core message is that integrating predictive and prescriptive models in a unified information system allows managers to test how preparedness budgets, pre‑positioning, and dispatch rules affect downstream wildfire losses, thus supporting more risk‑informed investment and response policies. [pubsonline.informs](https://pubsonline.informs.org/doi/10.1287/isre.2022.0118)

***

**10. Tezcan, B., & Eren, T. (2025). “Forest fire management and fire suppression strategies: a systematic literature review.” *Natural Hazards*, 121(9), 10485–10515. DOI: 10.1007/s11069-025-07227-x.** [ideas.repec](https://ideas.repec.org/a/spr/nathaz/v121y2025i9d10.1007_s11069-025-07227-x.html)

- **Scope & methods.** Reviews 92 optimization and decision-analytic studies on forest-fire resource management, including location–allocation, routing, stochastic programming, and simulation–optimization models for suppression planning and dispatch. [d-nb](https://d-nb.info/1367298660/34)
- **Decision-support relevance (paraphrased).** The authors explicitly conclude that combining simulation and optimization is central to building effective decision-support tools for wildfire resource allocation, but they also note a persistent gap between sophisticated models and operational adoption by agencies. [academia](https://www.academia.edu/128253415/Forest_fire_management_and_fire_suppression_strategies_a_systematic_literature_review?force_claim_to_highlight=true)

***

## Operational decision-support systems in practice

**11. Fillmore, S. D., & Paveglio, T. B. (2023). “Use of the Wildland Fire Decision Support System (WFDSS) for full suppression and managed fires within the Southwestern Region of the US Forest Service.” *International Journal of Wildland Fire*, 32, WF22206. DOI: 10.1071/WF22206.** [nwfirescience](https://nwfirescience.org/sites/default/files/publications/Fillmore%20et%20al_2023%20Use%20of%20WFDSS%20for%20full%20suppression%20and%20managed%20fires%20in%20SW%20region%20of%20USFS.pdf)

- **Scope & methods.** Qualitative study based on interviews with USFS fire managers about how they actually use WFDSS for both full-suppression and managed-fire incidents in the US Southwest. [nwfirescience](https://nwfirescience.org/sites/default/files/publications/Fillmore%20et%20al_2023%20Use%20of%20WFDSS%20for%20full%20suppression%20and%20managed%20fires%20in%20SW%20region%20of%20USFS.pdf)
- **Decision-support relevance (paraphrased).** Managers see WFDSS as most valuable for structuring decision processes, aggregating spatial and modelling outputs, and documenting rationale; the paper stresses that without attention to training, roles, and information overload, DSS outputs may end up primarily justifying intuition rather than shaping resource deployments. [nwfirescience](https://nwfirescience.org/sites/default/files/publications/Fillmore%20et%20al_2023%20Use%20of%20WFDSS%20for%20full%20suppression%20and%20managed%20fires%20in%20SW%20region%20of%20USFS.pdf)

***

**12. Epstein, M. D., & Seielstad, C. A. (2025). “Learning from Wildfire Decision Support: large language model analysis of barriers to fire spread in a census of large wildfires in the United States (2011–2023).” *International Journal of Wildland Fire*, 34, WF25051. DOI: 10.1071/WF25051.** [publish.csiro](https://www.publish.csiro.au/wf/pdf/WF25051)

- **Scope & methods.** Uses a large language model to analyze textual barrier descriptions in WFDSS records for 6,630 large US wildfires, extracting patterns in how decision-makers discuss and leverage barriers to fire spread. [publish.csiro](https://www.publish.csiro.au/wf/pdf/WF25051)
- **Decision-support relevance (paraphrased).** The study argues that LLM-based text mining can surface consistent, yet previously “hidden,” patterns in expert decision rationales, providing feedback that could improve future DSS design and training around barrier use and suppression strategy selection. [publish.csiro](https://www.publish.csiro.au/wf/pdf/WF25051)

***

**13. Beeton, T. A., Aldworth, T., Colavito, M. M., et al. (2025). “The Diffusion of Risk Management Assistance for Wildland Fire Management in the United States.” *Fire*, 8(12), 478. DOI: 10.3390/fire8120478.** [ui.adsabs.harvard](https://ui.adsabs.harvard.edu/abs/2025Fire....8..478B/abstract)

- **Scope & methods.** Examines how the US Forest Service’s Risk Management Assistance (RMA) program and related decision-support tools (e.g., PODs) are spreading through the wildland fire management system, using mixed methods on adoption and use. [nwfirescience](https://nwfirescience.org/research-database?f%5B0%5D=publication_keywords%3A681&f%5B1%5D=publication_keywords%3A1420&f%5B2%5D=year_of_publication%3A2025&field_publication_topics_target_id=5989)
- **Decision-support relevance (paraphrased).** The authors emphasize that diffusion of decision-support is shaped as much by organizational culture, trust, and perceived usefulness as by technical performance, and they frame RMA as a bridge between risk analytics and on-the-ground strategic and tactical choices. [eri.nau](https://eri.nau.edu/wp-content/uploads/2025/08/FY23_Annual-Report_FY23_FINAL_03-28-25_for_WEB.pdf)

***

## Tiered risk / response classification and temporal validation

**14. Caron, N., Noura, H., Guyeux, C., & Aynes, B. (2025). “A Voting System to Optimize Daily Forest Fire Prediction.” In *2025 IEEE 37th International Conference on Tools with Artificial Intelligence (ICTAI)*, 57–64. DOI: 10.1109/ICTAI66417.2025.00016.** [computer](https://www.computer.org/csdl/proceedings-article/ictai/2025/491900a057/2ct10xPSOCA)

- **Scope & methods.** Treats daily forest-fire risk as an ordinal five-class problem and proposes an ensemble “voting” scheme across several models; evaluates performance with F1 and Intersection over Union, with special attention to class imbalance and multi-class metrics. [hal](https://hal.science/hal-05532304/document)
- **Decision-support relevance (paraphrased).** They argue that multi-class (tiered) risk levels are more actionable for operational services than binary outputs, and that IoU-based evaluation better reflects the cost of confusing adjacent vs. distant risk tiers when risk maps are used to trigger different preparedness levels. [arxiv](https://arxiv.org/html/2506.04254v1)

***

**15. Caron, N., Guyeux, C., Noura, H., & Aynes, B. (2025). “Localized Forest Fire Risk Prediction: A Department-Aware Approach for Operational Decision Support.” arXiv:2506.04254.** [arxiv](https://arxiv.org/abs/2506.04254)

- **Scope & methods.** Builds a national-scale AI benchmark for France in which risk is predicted as a five-level ordinal variable (0–4) for each department and forecast horizon (1, 7, 15, 31 days), combining meteorology, fire indices, historical activity and land cover; compares multiple models including CatBoost and time-series architectures. [arxiv](https://arxiv.org/html/2506.04254v2)
- **Decision-support relevance (paraphrased).** The authors explicitly design their multi-class scheme to align with operational jurisdictions and to make “high” vs. “extreme” tiers directly interpretable by fire services, showing that multi-class models provide richer, more regionally meaningful inputs for tiered preparedness and resource allocation than conventional binary danger indices. [arxiv](https://arxiv.org/html/2506.04254v1)

***

**16. Lee, S.-L., Hsu, M.-H., Wang, Y.-F., & Wang, M. Y.-F. (2025). “Machine learning-based forecasting of urban fire impact in city environments.” *Scientific Programming* (Sci Prog) 108(4), 00368504251406566. DOI: 10.1177/00368504251406566.** [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12690061/)

- **Scope & methods.** Uses XGBoost and related ML models on 47,382 urban incidents to predict severity (major vs ordinary) from building, temporal, and GIS-derived features, with 5‑fold, temporal, and geographic validation to test generalization. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12690061/)
- **Decision-support relevance (paraphrased).** The paper explicitly proposes using severity predictions to drive tiered dispatch protocols (e.g., automatically larger initial responses for certain building types), arguing that such predictive tiering can reduce reliance on intuition and improve resource allocation during concurrent incidents. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12690061/)

***

**17. Anonymous (2025). “Forecasting urban fire severity for enhanced emergency response operations.” *Scientific Reports* 15, Article 26006. DOI: 10.1038/s41598-025-26006-z.** [nature](https://www.nature.com/articles/s41598-025-26006-z)

- **Scope & methods.** Develops an ML model (including XGBoost and baselines such as DT and RF) for multi-class urban fire severity prediction and evaluates with 5‑fold cross‑validation plus temporal and geographic holdouts, emphasizing robustness across time and space. [nature](https://www.nature.com/articles/s41598-025-26006-z)
- **Decision-support relevance (paraphrased).** The authors argue that rigorous temporal and spatial validation is essential before integrating severity forecasts into computer-aided dispatch systems, as naive random splits overestimate performance and could mislead resource planning. [nature](https://www.nature.com/articles/s41598-025-26006-z)

*(Although urban rather than wildland, these two papers are among the clearest exemplars of tiered-response classification and temporal validation protocols in fire dispatch ML, and their methodology is transferable to wildland settings.)*  

***

## Reviews and conceptual / methodological foundations

**18. Martell, D. L., et al. (2020). “A review of machine learning applications in wildfire science and management.” *Environmental Reviews*, 28(4), 1–21. DOI: 10.1139/er-2020-0019.** [cdnsciencepub](https://cdnsciencepub.com/doi/full-xml/10.1139/er-2020-0019)

- **Scope & methods.** Synthesizes pre‑2020 ML work across ignition, spread, risk, and management, including probability-of-initial-attack success models and emerging dispatch/containment modelling. [cdnsciencepub](https://cdnsciencepub.com/doi/10.1139/er-2020-0019)
- **Decision-support relevance (paraphrased).** The review concludes that ML is most impactful when embedded into broader decision-support frameworks that respect operational constraints, highlighting initial-attack success and resource optimization as promising but under‑implemented use cases. [cdnsciencepub](https://cdnsciencepub.com/doi/full-xml/10.1139/er-2020-0019)

***

**19. Abatzoglou, J. T., & Balch, J. K. (2018). “Human-related ignitions concurrent with high winds promote large wildfires across the USA.” *International Journal of Wildland Fire*, 27, 1–10. (Foundational concurrency work.)** [nwfirescience](https://www.nwfirescience.org/biblio/human-related-ignitions-concurrent-high-winds-promote-large-wildfires-across-usa)

- **Scope & methods.** Examines how human ignitions combined with concurrent high‑wind conditions drive large-fire occurrence across ecoregions, using comparative analyses of large vs small wildfires. [pyrogeographer](http://www.pyrogeographer.com/uploads/1/6/4/8/16481944/abatzoglou_etal_2018_ijwf.pdf)
- **Decision-support relevance (paraphrased).** The authors argue that incorporating wind and ignition-concurrency into large‑fire risk models is critical for prioritizing prevention and suppression resources, a principle that later ML-based IA and dispatch models operationalize via concurrent-fire and high-wind features. [nwfirescience](https://www.nwfirescience.org/biblio/human-related-ignitions-concurrent-high-winds-promote-large-wildfires-across-usa)

***

**20. Suarez, D., et al. (2024) and Tezcan & Eren (2025) – integration perspective.** [ideas.repec](https://ideas.repec.org/a/spr/nathaz/v121y2025i9d10.1007_s11069-025-07227-x.html)

- These works jointly argue that the most effective wildfire decision-support systems combine:  
  - probabilistic/ML risk models (for ignitions, escape probability, severity, aircraft targets),  
  - optimization and simulation models (for resource placement and dispatch), and  
  - socio-technical analysis of how tools like WFDSS, RMA, and risk indices are actually used by incident commanders. [ui.adsabs.harvard](https://ui.adsabs.harvard.edu/abs/2025Fire....8..478B/abstract)

***

## How these works address your four focal themes

- **Tier-based response classification.**  
  - Caron et al. (ICTAI 2025, arXiv 2025) explicitly treat wildfire risk as a five-level ordinal target and argue this better matches operational decision tiers than binary danger labels. [hal](https://hal.science/hal-05532304/document)
  - Urban severity papers (Sci Prog, Sci Rep) demonstrate multi-tier fire severity predictions feeding tiered dispatch recommendations, with rigorous multi-split validation. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12690061/)

- **Two-stage models for imbalanced aerial dispatch.**  
  - Taylor & Nadeem split modelling into fire-occurrence and conditional aircraft-use components, then combine them to estimate daily aircraft IA targets, explicitly addressing rarity of high-use days. [ouci.dntb.gov](https://ouci.dntb.gov.ua/en/works/lRjZPejl/)
  - Wheatley et al. show that standard models underpredict rare “many airtankers” dispatches, pointing to the need for specialized handling of these imbalanced classes, potentially via two-stage or cost-sensitive designs. [cdnsciencepub](https://cdnsciencepub.com/doi/full/10.1139/cjfr-2022-0225)
  - Suarez et al. provide a conceptual two-stage architecture: predictive risk models feeding prescriptive resource-allocation simulations for wildfire control. [research.sabanciuniv](https://research.sabanciuniv.edu/id/eprint/49782/)

- **Temporal validation protocols for wildfire ML.**  
  - Caron’s department-aware benchmark explicitly tests models across multiple forecast horizons and discusses how historical signals dominate at longer horizons, implicitly motivating horizon-specific evaluation. [arxiv](https://arxiv.org/html/2506.04254v2)
  - Urban severity works use combined k‑fold, temporal, and geographic validation and argue that temporal holdouts are essential before deployment within CAD systems. [nature](https://www.nature.com/articles/s41598-025-26006-z)
  - Several susceptibility and risk papers (e.g., Frontiers in Forests and Global Change 2025) use multi-year splits and SHAP analysis to ensure stability and interpretability of fire-risk tiers over time. [frontiersin](https://www.frontiersin.org/journals/forests-and-global-change/articles/10.3389/ffgc.2025.1705341/full)

- **Feature engineering with concurrent-fire signals.**  
  - The Catalonia IA study includes explicit “fire simultaneity” features (e.g., number of active fires in the last 24 h, simultaneous fire episodes with n > 10) that strongly influence escape probability. [zaguan.unizar](https://zaguan.unizar.es/record/112180/files/texto_completo.pdf)
  - Follow-on containment and landscape-planning work reuses these ideas, incorporating “number of active fires” and fire-weather combinations into spatial risk and containment models. [zaguan.unizar](https://zaguan.unizar.es/record/112180/files/texto_completo.pdf)
  - Department-aware risk models (Caron et al.) show that historical and short-term fire occurrence features dominate feature importance, capturing aspects of clustering and concurrency at departmental scale even without an explicit “simultaneous fires” variable. [arxiv](https://arxiv.org/html/2506.04254v1)


# Synthetic paragraphs for literature-review:

Here is a structured markdown literature‑review narrative synthesizing the works we discussed, with a focus on methodological typology and gaps.

***

## Scope and problem framing

Research on wildfire dispatch prediction and resource allocation has evolved from isolated probabilistic models of initial‑attack success to richer combinations of machine learning, optimization, and socio‑technical decision-support systems. [cdnsciencepub](https://cdnsciencepub.com/doi/10.1139/er-2020-0019)
Within 2019–2025, four strands stand out: (i) predictive models for initial‑attack outcomes and aircraft dispatch, (ii) optimization and metaheuristics for resource placement and dispatch, (iii) tier‑based risk and severity classification, and (iv) studies of how tools like WFDSS and Risk Management Assistance are adopted in practice. [nwfirescience](https://nwfirescience.org/sites/default/files/publications/Fillmore%20et%20al_2023%20Use%20of%20WFDSS%20for%20full%20suppression%20and%20managed%20fires%20in%20SW%20region%20of%20USFS.pdf)

***

## Predictive models for initial‑attack and aircraft dispatch

Early work on initial‑attack (IA) success framed containment as a binary outcome and used random forests and related ML methods to estimate success probabilities conditioned on weather, access, resources, and landscape context. [arxiv](https://arxiv.org/pdf/2003.00646.pdf)
Rodrigues et al. in Catalonia show that accessibility and aerial support strongly improve IA success, and they explicitly include “fire simultaneity” as a feature, linking concurrent ignitions and high‑demand periods to reduced containment probabilities. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0048969719308319)

Subsequent studies refine this approach by separating ignition and suppression decisions, and by extending the outcome space beyond simple success/failure.  
Taylor and Nadeem, for example, first model daily ignition probabilities and then, conditional on ignitions, model whether aircraft are used on initial attack, effectively creating a two‑stage pipeline for predicting “aircraft IA targets” across the landscape. [ouci.dntb.gov](https://ouci.dntb.gov.ua/en/works/lRjZPejl/)
Wheatley et al. go further by modelling multi‑class airtanker dispatch (0, 1, 2, or more large airtankers), effectively capturing a tiered resource‑use decision and revealing systematic underprediction of rare “many‑airtanker” events under standard statistical models. [cdnsciencepub](https://cdnsciencepub.com/doi/full/10.1139/cjfr-2022-0225)

More recent work proposes indices designed to plug directly into operational decision-support systems.  
Cardil et al.’s Initial Attack Assessment (IAA) index, derived from large samples of California ignitions and simulation outputs, condenses suppression difficulty into a 1–5 scale while linking it statistically to IA success probabilities via GLMs that incorporate fire behaviour, terrain difficulty, and response time. [nwfirescience](https://www.nwfirescience.org/sites/default/files/publications/wf24160.pdf)
Alawode et al. complement these global indices with spatially explicit containment probability maps for escaped fires, using random forests to estimate where suppression is more or less likely to succeed at fine spatial resolution. [zaguan.unizar](https://zaguan.unizar.es/record/148652/files/texto_completo.pdf)

***

## Optimization and metaheuristics for resource placement and dispatch

Parallel to predictive modelling, a longstanding line of research formulates resource placement and dispatch as optimization problems.  
Mendes and Alvelos develop a mixed‑integer programming model for locating wildland fire suppression resources over a grid and solve it via iterated local search, treating ignition scenarios as demand points and optimising coverage and impact metrics subject to resource constraints. [ideas.repec](https://ideas.repec.org/a/eee/ejores/v304y2023i3p887-900.html)

Subsequent work by Matos, Rocha, and colleagues shifts the focus from static placement to dynamic dispatch, modelling the assignment of multiple firefighting resources to sets of ignitions as a combinatorial optimisation problem solved with genetic algorithms. [repositorium.uminho](https://repositorium.uminho.pt/bitstreams/593d9440-df95-4fa3-b211-aab0397ae18e/download)
These studies emphasise that metaheuristics can explore large dispatch decision spaces more effectively than manual heuristics, especially under multi‑fire, resource‑scarce conditions, and they explicitly present their frameworks as decision-support tools to test alternative dispatch strategies before or during fire seasons. [semanticscholar](https://www.semanticscholar.org/paper/Resource-Dispatch-Optimization-for-Firefighting-on-Matos-Rocha/00ce4f4a31c2e34dd96a89cae987e08b6892cad5)

Broader reviews and methodological papers underscore that optimisation is most powerful when integrated with predictive and simulation components.  
Suarez et al. propose an “integrated decision support” architecture in which probabilistic risk models feed into optimisation and simulation of preparedness and response policies, using wildfire management in Uruguay as a primary case. [pubsonline.informs](https://pubsonline.informs.org/doi/10.1287/isre.2022.0118)
Tezcan and Eren’s systematic review similarly catalogues location–allocation, vehicle-routing and stochastic programming models for forest fire suppression, and highlights the persistent gap between sophisticated optimisation models and actual adoption in agency decision processes. [ideas.repec](https://ideas.repec.org/a/spr/nathaz/v121y2025i9d10.1007_s11069-025-07227-x.html)

***

## Tier-based risk and severity classification

A distinct methodological strand focuses on translating continuous or probabilistic risk into discrete tiers that correspond more directly to operational response levels.  
Caron and colleagues cast daily forest‑fire risk prediction as a five‑class ordinal problem and implement an ensemble “voting system” where multiple classifiers’ outputs are combined to produce daily risk maps evaluated with F1 and Intersection‑over‑Union metrics. [hal](https://hal.science/hal-05532304/document)
They argue that multi‑class, ordinal risk maps better align with how agencies define preparedness levels than binary “fire/no‑fire” predictions, and that metrics like IoU reflect the operational cost of confusing adjacent vs distant risk categories. [computer](https://www.computer.org/csdl/proceedings/ictai/2025/2ct0EAiUeg8)

Building on this, the “department‑aware” risk benchmark for France extends tiered risk prediction in both space and time, predicting 0–4 risk levels across departments for several forecast horizons using gradient boosting and other ML models. [arxiv](https://arxiv.org/abs/2506.04254)
The authors explicitly co‑design the target space with fire services so that model outputs can be plugged into existing multi‑level preparedness protocols, and they demonstrate that longer horizons are dominated by historical and structural features, whereas shorter horizons gain more from high‑resolution meteorological inputs. [arxiv](https://arxiv.org/html/2506.04254v1)

Although focused on urban rather than wildland fires, recent works on urban fire severity prediction illustrate mature methodological practices likely to influence wildland dispatch modelling.  
These studies use multi‑class severity outcomes and evaluate models under k‑fold, temporal, and geographic cross‑validation, and they position severity forecasts as direct inputs to tiered dispatch rules in computer‑aided dispatch systems. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12690061/)

***

## Decision-support systems and socio-technical context

Studies of actual decision-support systems underscore that analytic sophistication is only one component of effective wildfire decision‑making.  
Fillmore and Paveglio’s qualitative work on the Wildland Fire Decision Support System (WFDSS) shows that managers value the system more for structuring decisions, aggregating spatial and model outputs, and documenting rationale than for any specific predictive algorithm, and it identifies training and usability as critical bottlenecks. [discovery.researcher](https://discovery.researcher.life/article/use-of-the-wildland-fire-decision-support-system-wfdss-for-full-suppression-and-managed-fires-within-the-southwestern-region-of-the-us-forest-service/c3cc87745f4c30da92fc333ce3e40ca0)

Beeton and colleagues’ analysis of the Risk Management Assistance (RMA) program similarly examines how risk analytics and pre‑fire planning tools (e.g., Potential Operational Delineations) diffuse through the US wildland fire management system. [cfri.colostate](https://cfri.colostate.edu/wp-content/uploads/sites/22/2025/12/2025_DST_FIRE_Beetonetal.pdf)
They highlight that adoption depends on organisational trust, alignment with existing workflows, and perceived usefulness in supporting strategic and tactical decisions, and they document efforts to integrate RMA outputs into WFDSS workflows. [wildfire](http://www.wildfire.gov/application/wfdss)

Complementary work uses large language models to mine textual decision records and barrier descriptions in WFDSS, aiming to learn from the “soft” rationales behind spread‑limiting strategies.  
This line of research suggests that decision-support research is increasingly concerned with closing feedback loops between model outputs, human judgment, and post‑incident analysis, not just improving predictive accuracy in isolation. [ll.mit](https://www.ll.mit.edu/media/6846)

***

## Cross-cutting methodological themes

### Temporal validation and generalization

A recurring concern is how to evaluate wildfire ML models so that they generalize across fire seasons and changing climate regimes.  
Many earlier IA‑success and susceptibility studies rely on random or spatial cross‑validation, which can overstate performance when temporal autocorrelation or regime shifts are present. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/30818214/)

More recent tiered risk and severity models explicitly use temporal and geographic validation schemes.  
Caron’s department‑aware benchmark, for example, assesses performance at multiple forecast horizons and tests generalization across years and regions, while urban fire severity papers use temporal holdouts and distinct geographic splits to approximate real‑world deployment conditions. [arxiv](https://arxiv.org/html/2506.04254v2)
Nevertheless, comparable temporal validation protocols are still relatively rare in wildland aerial dispatch and IA‑success studies, where models are often validated on pooled historical data without explicit time‑based splits. [ouci.dntb.gov](https://ouci.dntb.gov.ua/en/works/lDXGgezl/)

### Feature engineering and concurrent-fire signals

Feature engineering choices strongly influence how models capture operational stress and resource scarcity.  
Rodrigues et al. explicitly include “fire simultaneity” indicators (e.g., number of simultaneous ignitions exceeding certain thresholds) and show that higher simultaneity is associated with lower IA success, reflecting the reality that resources are stretched when many fires start at once. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0048969719308319)

Later work on containment probability and landscape planning incorporates related ideas by combining top-down risk indices, local fire weather, and measures of recent fire activity or ignition clustering. [zaguan.unizar](https://zaguan.unizar.es/record/112180/files/texto_completo.pdf)
Caron’s department‑level risk models, while not explicitly labelling a “concurrent fires” feature, effectively encode concurrency through recent-fire and historical activity variables at the department scale, which dominate feature importance at longer forecast horizons. [arxiv](https://arxiv.org/html/2506.04254v1)

Despite these examples, explicit modelling of fire queues, system load, or dispatch–demand coupling remains underdeveloped in most ML‑based IA and dispatch models.  
In practice, the interaction between forecasted risk, available resources, and the probability that some fires will be held with fewer resources than ideal is often left implicit, limiting the value of risk forecasts for true dispatch optimization. [ntrs.nasa](https://ntrs.nasa.gov/api/citations/20240015828/downloads/Wildfire%20Simulation%20Optimization.pdf)

### Class imbalance and rare high-consequence events

Airtanker dispatch and large‑escape events are quintessentially imbalanced: most fires require no aircraft or small ground responses, while a small subset accounts for extreme resource use and damage.  
Wheatley’s analysis highlights that standard models fit frequent low‑resource classes well but underpredict rare high‑dispatch decisions, and that this mismatch can reflect both statistical limitations and risk‑averse human behaviour. [cdnsciencepub](https://cdnsciencepub.com/doi/full/10.1139/cjfr-2022-0225)

Caron’s multi‑class frameworks adopt ordinal loss functions and metrics like IoU to better reflect the cost of misclassifying high‑risk classes, although they still rely primarily on reweighting and ensemble methods rather than problem‑specific two‑stage or anomaly‑detection schemes. [computer](https://www.computer.org/csdl/proceedings-article/ictai/2025/491900a057/2ct10xPSOCA)
Urban severity models similarly note that classes corresponding to very severe fires are underrepresented and call for cost‑sensitive training and evaluation tailored to dispatch decision thresholds. [nature](https://www.nature.com/articles/s41598-025-26006-z)

In sum, while imbalance is widely recognized, there is limited work on explicitly two‑stage designs tuned to “rare but critical” initial‑attack and dispatch outcomes—e.g., using a first‑stage model for overall risk and a second specialised model for extreme demand days or multi‑aircraft deployments.

***

## Methodological gaps and research opportunities

### Limited end-to-end integration of predictive and prescriptive models

Most studies either estimate risk/IA success (predictive) or optimize resource placement/dispatch (prescriptive), but relatively few build integrated pipelines where uncertainty-aware predictions drive optimization under dynamic resource constraints. [research.sabanciuniv](https://research.sabanciuniv.edu/id/eprint/49782/)
Suarez et al.’s disaster-risk framework and NASA’s simulation‑based asset‑location work show the feasibility of coupling ML, simulation, and optimisation, yet such architectures are not yet standard in wildland fire operations. [ntrs.nasa](https://ntrs.nasa.gov/api/citations/20240015828/downloads/Wildfire%20Simulation%20Optimization.pdf)

**Opportunity:** design modular, end‑to‑end decision-support systems where ML‑based risk indices, IA‑success probabilities, and concurrent‑fire features feed simulation–optimization components that output actionable pre‑positioning and dispatch recommendations under real-time constraints.

### Underdeveloped temporal robustness in wildland IA and dispatch models

While tiered risk and urban severity papers adopt rigorous temporal validation, wildland IA‑success and airtanker-dispatch models still largely rely on pooled cross‑validation, which may not reflect changing fuel, climate, and operational regimes. [cdnsciencepub](https://cdnsciencepub.com/doi/10.1139/er-2020-0019)
This is especially problematic as many agencies need tools that remain credible as climate‑driven extremes push fires outside historical envelopes. [wfmrda.nwcg](http://wfmrda.nwcg.gov/newsletter/wfm-rda-newsletter-march-2023)

**Opportunity:** adopt standardised temporal and spatio‑temporal validation protocols (e.g., rolling-origin or blocked cross‑validation) for wildland IA and dispatch models, alongside explicit assessment of performance stability across fire‑season types and climate anomalies.

### Sparse explicit modelling of concurrent-fire load and queues

Only a handful of IA‑success models explicitly encode fire simultaneity, and even fewer attempt to model the interaction of simultaneous fires, resource queues, and response times in a coupled way. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/30818214/)
Optimization studies often assume fixed scenario sets or static resource capacities, treating each scenario independently rather than within a system experiencing multiple concurrent incidents. [fs.usda](https://www.fs.usda.gov/rm/pubs_other/rmrs_2011_masarie_a001.pdf)

**Opportunity:** combine ML-based risk forecasts with queueing or simulation frameworks that explicitly represent simultaneous fires, resource travel times, and potential backlogs, using derived features (e.g., expected number of IA targets, distribution of risk among incidents) as inputs to dispatch policies.

### Limited embedding of tiered models into explicit dispatch rules

Although several studies advocate tiered risk or severity schemes, few specify or test concrete policy mappings from risk/severity tiers to dispatch levels (e.g., resource packages per tier) within realistic constraints. [hal](https://hal.science/hal-05532304/document)
Taylor’s aircraft IA targets and Cardil’s IAA index come close, but even there the link between the index and explicit policy actions is largely conceptual rather than formally optimised and evaluated. [nwfirescience](https://www.nwfirescience.org/biblio/assessing-suppression-difficulty-wildland-fires-initial-attack-response)

**Opportunity:** co‑design tiered risk indices and dispatch/response policies, then test them via simulation and counterfactual analysis, assessing both operational cost and outcome metrics (e.g., probability of escape, resource utilisation) under realistic incident streams.

### Socio-technical adoption and human–AI collaboration

DSS adoption studies emphasise that technical performance is only one driver of real-world impact; training, interface design, trust, and institutional incentives are equally important. [ui.adsabs.harvard](https://ui.adsabs.harvard.edu/abs/2025Fire....8..478B/abstract)
Fillmore and Paveglio show that WFDSS is appreciated for structuring decisions but can overwhelm users with information, while RMA diffusion studies reveal the importance of aligning new analytics with existing workflows and decision cultures. [nwfirescience](https://nwfirescience.org/sites/default/files/publications/Fillmore%20et%20al_2023%20Use%20of%20WFDSS%20for%20full%20suppression%20and%20managed%20fires%20in%20SW%20region%20of%20USFS.pdf)

**Opportunity:** embed ML and optimisation models inside human‑centred DSS workflows, with iterative co‑design, explainability, and post‑incident learning loops—potentially leveraging LLMs to summarise rationales, compare model outputs with decisions, and suggest training needs.

### Benchmarking and data availability

The lack of open, standardised benchmarks for dispatch and IA‑success modelling limits reproducibility and comparison across methods.  
Caron’s France benchmark is an important step, providing multi‑year, multi‑horizon, department‑level risk data and baseline models, but similar resources are rare for aerial dispatch or IA‑success in other jurisdictions. [arxiv](https://arxiv.org/pdf/2003.00646.pdf)

**Opportunity:** create open, standardised benchmark datasets for IA success, aerial dispatch, and concurrent‑fire load, with clearly defined targets, features, and temporal splits, akin to widely-used benchmarks in other ML domains.  

***



