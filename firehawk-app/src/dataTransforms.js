// dataTransforms.js

// Aerial activation threshold used by the pipeline's two-stage aerial model.
export const AERIAL_THRESHOLD = 0.35;

// Tier index -> human label (Tier 0 = smallest deployment, Tier 2 = largest).
export const TIER_LABELS = ['Minimal', 'Standard', 'Reinforced'];

/** Resolve a tier label from a numeric tier (falls back to the raw label). */
export function tierLabelOf(tier, rawLabel) {
  if (rawLabel) return rawLabel;
  if (tier == null) return null;
  return TIER_LABELS[tier] ?? null;
}

/**
 * Human-readable aerial-support indicator.
 * prob >= threshold -> "Likely (0.41)" (+ " · ~N" when an expected count exists);
 * otherwise -> "Unlikely". Returns "N/A" when no probability is available.
 */
export function formatAerial(prob, expected) {
  if (prob == null) return 'N/A';
  if (prob >= AERIAL_THRESHOLD) {
    const p = Number(prob).toFixed(2);
    return expected != null ? `Likely (${p}) · ~${expected}` : `Likely (${p})`;
  }
  return 'Unlikely';
}

export function transformFireData(jsonData) {
  if (!Array.isArray(jsonData)) {
    return [];
  }

  // 1
  return jsonData.map((fire, index) => {
    // --- ADAPTATION 1: DATE ---
    // The new pipeline sends 'data' and 'hora'. The old one sent 'Data_Atualizacao'.
    let formattedDate = 'N/A';

    if (fire.data && fire.hora) {
      // The new format coming from Python
      const dateFixed = fire.data.replace(/-/g, '/');
      formattedDate = `${dateFixed}, ${fire.hora}`;
    } else if (fire.Data_Atualizacao) {
      // Keep your original fallback logic
      const updateDate = new Date(fire.Data_Atualizacao.replace(' ', 'T'));
      if (!isNaN(updateDate.getTime())) {
        formattedDate = updateDate.toLocaleString('pt-PT', {
          day: '2-digit',
          month: '2-digit',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        });
      }
    }

    // --- ADAPTATION 2: RESOURCES ---
    // Map the new fields (Real_...) or use the old ones (Operacionais_...) as fallback
    const man = Number(fire.Real_Homens) || Number(fire.Operacionais_Man) || 0;
    const terrain = Number(fire.Real_Terrestres) || Number(fire.Meios_Terrestres) || 0;
    const aerial = Number(fire.Real_Aereos) || Number(fire.Real_Meios_Aereos) || Number(fire.Meios_Aereos) || 0;

    // Map the new fields (Previsto_...) or use the old ones (Prev_Homens, etc.) as fallback
    const predictedMan = Number(fire.Previsto_Operacionais_Man) || Number(fire.Prev_Homens) || 0;
    const predictedTerrain = Number(fire.Previsto_Meios_Terrestres) || Number(fire.Prev_Terrestres) || 0;
    const predictedAerial = Number(fire.Previsto_Meios_Aereos) || Number(fire.Prev_Aereos) || 0;

    // Adjustment to display the full string on the Dashboard (H | T | A) now that we have these data
    const units = `${man} M | ${terrain} T | ${aerial} A`;

    // --- ADAPTATION 2b: TIER FORECAST (new schema) ---
    // Use nullish checks so tier 0 and aerial_prob 0 survive. Old historical files
    // lack these fields, so they resolve to null (UI shows N/A) — no derivation.
    const tier = fire.tier != null ? Number(fire.tier) : null;
    const tierLabel = tierLabelOf(tier, fire.tier_label);
    const opsRange = fire.ops_range ?? null;
    const vehRange = fire.veh_range ?? null;
    const aerialProb = fire.aerial_prob != null ? Number(fire.aerial_prob) : null;
    const aerialExpected = fire.aerial_expected != null ? Number(fire.aerial_expected) : null;

    // --- ADAPTATION 3: STATUS / LEVEL ---
    // Map 'status' (new) or 'Estado' (old)
    const currentStatus = fire.status || fire.Estado;
    let level = 0;
    
    // Keeping your switch case, adding only fallback for common Python statuses
    switch (currentStatus) {
      case 'Despacho de 1º Alerta':
      case 'Despacho': 
        level = 1;
        break;
      case 'Em Resolução':
      case 'Em Curso': 
      case 'Ativo':  
      case 'Chegada ao TO':  
        level = 2;
        break;
      case 'Vigilância':
        level = 3;
        break;
      case 'Conclusão':
        level = 4;
        break;
      default:
        level = 0;
    }

    // --- ADAPTATION 4: LOCATION ---
    // The pipeline sends 'local'. If absent, use your old logic.
    const location = fire.local || 
      (fire.Distrito && fire.Concelho
        ? `${fire.Distrito} - ${fire.Concelho}`
        : fire.Distrito || fire.Concelho || fire.Localizacao || 'Unknown');

    // --- ADAPTATION 5: ENRICHED DATA (For AlertDetails) ---
    const enrichedOriginal = {
      ...fire,
      // Coordinate unification
      lat: parseFloat(fire.lat || fire.Latitude),
      lng: parseFloat(fire.lon || fire.lng || fire.Longitude),
      
      // Actual resources
      man,
      terrain,
      heliFight: aerial,

      // Metadata
      natureza: fire.natureza || fire.Natureza,
      regiao: fire.local || fire.Distrito,
      sub_regiao: fire.Concelho,

      // NEW WEATHER DATA (Requested for AlertDetails)
      pressao: fire.pressao,             // From Python
      direcao_vento: fire.direcao_vento, // From Python
      chuva_24h: fire.chuva_24h,  
      VPD_kPa: fire.vpd_kpa || fire.VPD_kPa || fire.vpd,      // From Python

      
      altitude: fire.altitude || fire.ALTITUDEMEDIA,
      fontealerta: fire.id || fire.NCCO, 
      fwi: fire.fwi || fire.FWI, 
      isi: fire.isi || fire.ISI,
      
      
      // FORECASTS (New pipeline fields for the comparison chart)
      Previsto_Operacionais_Man: predictedMan,
      Previsto_Meios_Terrestres: predictedTerrain,
      Previsto_Meios_Aereos: predictedAerial,
      
      // ACTUALS (Explicit for comparison)
      Real_Operacionais_Man: fire.Real_Homens,
      Real_Meios_Terrestres: fire.Real_Terrestres,
      Real_Meios_Aereos: fire.Real_Aereos,

      // TIER FORECAST (new schema, normalized)
      tier,
      tier_label: tierLabel,
      ops_range: opsRange,
      veh_range: vehRange,
      aerial_prob: aerialProb,
      aerial_expected: aerialExpected,
    };

    return {
      // internal incremental ID
      id: index + 1,
      // original incident ID
      originalId: String(fire.id || fire.ID_Incidente || fire.NCCO),
      lastlyUpdated: formattedDate,
      location,
      units,
      level,
      status: currentStatus || 'Unknown',
      // Tier forecast exposed on the row for the listing table
      tier,
      tier_label: tierLabel,
      ops_range: opsRange,
      veh_range: vehRange,
      aerial_prob: aerialProb,
      aerial_expected: aerialExpected,
      originalData: enrichedOriginal,
    };
  });
}