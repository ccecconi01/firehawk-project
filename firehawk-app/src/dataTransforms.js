// dataTransforms.js
export function transformFireData(jsonData) {
  if (!Array.isArray(jsonData)) {
    return [];
  }

  return jsonData.map((fire, index) => {
    // Use Data_Atualizacao as the "last updated" / reference date
    let formattedDate = 'N/A';
    if (fire.Data_Atualizacao) {
      // "2022-08-25 11:21:46" → valid Date string
      // The date format in the merged JSON is 'YYYY-MM-DD HH:MM:SS'
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

    // Resources 
    const man = Number(fire.Operacionais_Man) ||"N/A";
    const terrain = Number(fire.Meios_Terrestres) || "N/A";
    const units = man > 0 ? man : 'N/A';

    // Simple mapping from Estado to a numeric "level"
    // The original code used Estado_Descricao, which is now just Estado in the merged data.
    let level = 0;
    switch (fire.Estado) {
      case 'Despacho de 1º Alerta':
        level = 1;
        break;
      case 'Em Resolução':
        level = 2;
        break;
      case 'Vigilância':
        level = 2;
        break;
      case 'Conclusão':
        level = 3;
        break;
      default:
        level = 0;
    }

    // Human-readable location
    const location =
      fire.Distrito && fire.Concelho
        ? `${fire.Distrito} - ${fire.Concelho}`
        : fire.Distrito || fire.Concelho || fire.Localizacao || 'Unknown';

    // Enrich original data so AlertDetails can still use .lat/.lng/.man/.terrain
    const enrichedOriginal = {
      ...fire,
      // unified names the components expect
      lat: fire.Latitude,
      lng: fire.Longitude,
      man,
      terrain,
      heliFight: Number(fire.Real_Meios_Aereos) || Number(fire.Meios_Aereos) || 0,
      natureza: fire.Natureza, // Changed from Natureza_Descricao to Natureza
      regiao: fire.Distrito,
      sub_regiao: fire.Concelho,
      icnf: {
        altitude: fire.ALTITUDEMEDIA, // Added ALTITUDEMEDIA
        fontealerta: fire.NCCO, // Using NCCO as a source identifier
        fogacho: fire.FFMC, // Using FFMC as a fire weather index component
      },
      // Adding the RF prediction fields for AlertDetails to use
      Previsto_Operacionais_Man: fire.Previsto_Operacionais_Man,
      Erro_Operacionais_Man: fire.Erro_Operacionais_Man,
      Previsto_Meios_Terrestres: fire.Previsto_Meios_Terrestres,
      Erro_Meios_Terrestres: fire.Erro_Meios_Terrestres,
      Previsto_Meios_Aereos: fire.Previsto_Meios_Aereos,
      Erro_Meios_Aereos: fire.Erro_Meios_Aereos,
    };

    return {
      // internal incremental ID for the table / routing
      id: index + 1,
      // original incident ID so you can always refer back  
      originalId: fire.ID_Incidente, // ID_Incidente is the common key
      lastlyUpdated: formattedDate,
      location,
      units,
      level,
      status: fire.Estado || 'Unknown', // Changed from Estado_Descricao to Estado
      originalData: enrichedOriginal,
    };
  });
}