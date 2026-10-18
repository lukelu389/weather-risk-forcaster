# Financial context: from weather to an issuer decision

The project's CAD basket includes Fortis (FTS.TO), making it a useful example of the distinction between portfolio market risk and business exposure. The pilot estimates return-tail risk; it does not forecast Fortis revenue, cash flow or valuation.

## A primary-source mechanism

Fortis's 2025 MD&A attributes part of lower UNS Energy earnings to reduced retail electricity sales in milder weather. It also explains that regulatory revenue decoupling means sales changes at Central Hudson do not materially affect earnings. The same weather-to-volume shock can therefore have different earnings consequences across subsidiaries. See the [2025 MD&A, UNS Energy and Central Hudson sections, pp. 7–8](https://www.sec.gov/Archives/edgar/data/1666175/000166617526000013/a2025annual-993mda.htm).

Fortis identifies weather-related demand variability and notes that regulatory deferrals and decoupling at certain utilities reduce resulting earnings volatility. This supports checking jurisdiction-specific recovery mechanisms before translating degree days into an issuer sensitivity. See [2025 MD&A, Weather Variability and Seasonality, p. 25](https://www.sec.gov/Archives/edgar/data/1666175/000166617526000013/a2025annual-993mda.htm). These disclosures are external context, not model-training inputs or validation of the pilot.

## A diligence bridge, not a fitted valuation

An analyst could organize a scenario as:

`territory weather → electricity/gas demand → retained margin → regulatory recovery and cash timing → cash-flow / credit sensitivity`

This is an analytical interpretation of the mechanism above. It would require service-territory load, customer mix, tariffs, net retained margins, regulatory rules, costs, cash-collection timing and financing information. Toronto/Calgary temperatures and aggregate equity returns do not supply those inputs.

For a **purely hypothetical** volume-exposed business, assume baseline sales of 1,000 GWh, a 3% weather-related decline and CAD 20/MWh of retained margin. The isolated margin effect would be:

`−30 GWh × 1,000 MWh/GWh × CAD 20/MWh = −CAD 600,000`

This is arithmetic using invented illustration inputs, not a Fortis estimate or a result of this repository's model. It precedes taxes, operating responses, working capital and regulatory recovery. Applying the same calculation to a decoupled business without its recovery mechanics would misstate the earnings mechanism.

## How a banker or risk analyst would use the work

The useful decision is whether the evidence supports more targeted diligence. A banker could request subsidiary-level weather normalization and recovery schedules before including a downside case in financing or valuation work. A portfolio risk analyst could compare alternative risk forecasts while retaining calibration concerns. Neither role should substitute the current portfolio ES for an issuer cash-flow forecast or approved capital limit.

The pilot's lack of a general weather-feature advantage does not disprove the business channel. It tests a particular feature set, target and time period. A targeted issuer model would be a separate research question, requiring new data and a new evaluation design; it is outside this release's completed empirical claims.
