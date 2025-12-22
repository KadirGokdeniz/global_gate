import { AirlinePreference, Language } from '@/types';

interface AirlineSelectorProps {
  selectedAirline: AirlinePreference;
  onAirlineSelect: (airline: AirlinePreference) => void;
  language: Language;
}

export const AirlineSelector = ({ selectedAirline, onAirlineSelect, language }: AirlineSelectorProps) => {
  // ✅ "All Airlines" seçeneği kaldırıldı - Sadece THY ve Pegasus
  const airlines = [
    {
      id: 'thy' as AirlinePreference,
      icon: '🇹🇷',
      name: language === 'en' ? 'Turkish Airlines' : 'Türk Hava Yolları',
      description: language === 'en' ? 'THY baggage policies' : 'THY bagaj politikaları',
      gradient: 'from-red-600 to-red-700',
      bgColor: 'bg-red-50 dark:bg-red-900/20'
    },
    {
      id: 'pegasus' as AirlinePreference,
      icon: '✈️',
      name: language === 'en' ? 'Pegasus Airlines' : 'Pegasus Hava Yolları', 
      description: language === 'en' ? 'Pegasus baggage policies' : 'Pegasus bagaj politikaları',
      gradient: 'from-yellow-500 to-orange-600',
      bgColor: 'bg-yellow-50 dark:bg-yellow-900/20'
    }
  ];

  return (
    <div className="w-full max-w-3xl mx-auto mb-8">
      {/* Section Header */}
      <div className="text-center mb-6">
        <h3 className="text-lg font-semibold text-slate-700 dark:text-slate-200 mb-2">
          {language === 'en' ? 'Select Airline' : 'Havayolu Seçin'}
        </h3>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          {language === 'en' 
            ? 'Each airline has its own conversation history' 
            : 'Her havayolunun kendi konuşma geçmişi vardır'}
        </p>
      </div>

      {/* Airline Cards - 2 Column Grid (All Airlines kaldırıldı) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
        {airlines.map((airline) => (
          <button
            key={airline.id}
            onClick={() => onAirlineSelect(airline.id)}
            className={`group relative overflow-hidden rounded-2xl p-6 transition-all duration-500 hover:scale-[1.02] hover:shadow-2xl ${
              selectedAirline === airline.id
                ? 'shadow-2xl ring-2 ring-blue-500 ring-offset-2 ring-offset-white dark:ring-offset-slate-900'
                : 'shadow-lg hover:shadow-xl'
            }`}
          >
            {/* Background with gradient */}
            <div className={`absolute inset-0 bg-gradient-to-br ${airline.gradient} opacity-5 group-hover:opacity-10 transition-opacity duration-300`}></div>
            
            {/* Glass morphism background */}
            <div className={`absolute inset-0 backdrop-blur-sm transition-all duration-300 ${
              selectedAirline === airline.id
                ? 'bg-white/90 dark:bg-slate-800/90'
                : 'bg-white/70 dark:bg-slate-800/70 group-hover:bg-white/80 dark:group-hover:bg-slate-800/80'
            }`}></div>

            {/* Content */}
            <div className="relative z-10 flex flex-col items-center text-center space-y-4">
              {/* Icon with animation */}
              <div className={`text-5xl transition-transform duration-300 ${
                selectedAirline === airline.id ? 'scale-110' : 'group-hover:scale-110'
              }`}>
                {airline.icon}
              </div>

              {/* Airline Name */}
              <div>
                <div className={`font-bold text-xl transition-colors duration-300 ${
                  selectedAirline === airline.id
                    ? 'text-blue-700 dark:text-blue-300'
                    : 'text-slate-800 dark:text-slate-100 group-hover:text-slate-900 dark:group-hover:text-slate-50'
                }`}>
                  {airline.name}
                </div>

                {/* Description */}
                <div className={`text-sm font-medium mt-2 transition-colors duration-300 ${
                  selectedAirline === airline.id
                    ? 'text-blue-600 dark:text-blue-400'
                    : 'text-slate-600 dark:text-slate-400 group-hover:text-slate-700 dark:group-hover:text-slate-300'
                }`}>
                  {airline.description}
                </div>
              </div>

              {/* Selection Indicator */}
              <div className={`w-3 h-3 rounded-full transition-all duration-300 ${
                selectedAirline === airline.id
                  ? 'bg-blue-500 shadow-lg shadow-blue-500/50'
                  : 'bg-slate-300 dark:bg-slate-600 group-hover:bg-slate-400 dark:group-hover:bg-slate-500'
              }`}></div>
            </div>

            {/* Selected border effect */}
            {selectedAirline === airline.id && (
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-blue-500/20 to-indigo-500/20 animate-pulse"></div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
};