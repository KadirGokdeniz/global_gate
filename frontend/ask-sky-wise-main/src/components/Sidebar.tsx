import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { RefreshCw, Trash2, Zap, AlertCircle } from 'lucide-react';
import { Language, SessionStats, APIConnection } from '@/types';

interface SidebarProps {
  language: Language;
  t: (key: string) => string;
  apiConnection: APIConnection;
  sessionStats: SessionStats;
  onReconnect: () => void;
  onClearHistory: () => void;
}

export const Sidebar = ({
  language,
  t,
  apiConnection,
  sessionStats,
  onReconnect,
  onClearHistory
}: SidebarProps) => {
  const getConnectionStatus = () => {
    if (apiConnection.success) {
      return apiConnection.models_ready ? (
        <div className="status-success rounded-lg p-3 flex items-center gap-2">
          <Zap className="w-4 h-4" />
          <span className="text-sm font-medium">{t('apiConnected')} (Models Ready)</span>
        </div>
      ) : (
        <div className="status-warning rounded-lg p-3 flex items-center gap-2">
          <RefreshCw className="w-4 h-4 animate-spin" />
          <span className="text-sm font-medium">{t('apiConnected')} (Loading...)</span>
        </div>
      );
    }
    
    return (
      <div className="status-error rounded-lg p-3 flex items-center gap-2">
        <AlertCircle className="w-4 h-4" />
        <span className="text-sm font-medium">
          {t('apiFailed')}: {apiConnection.error}
        </span>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            🔗 {language === 'en' ? 'Connection Status' : 'Bağlantı Durumu'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {getConnectionStatus()}
          <Button 
            variant="outline" 
            onClick={onReconnect}
            className="w-full"
            size="sm"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            {t('reconnect')}
          </Button>
        </CardContent>
      </Card>

      {/* Session Statistics */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-lg">📊 {t('sessionStats')}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-3 bg-muted/30 rounded-lg">
              <div className="text-2xl font-bold text-primary">
                {sessionStats.totalQueries}
              </div>
              <div className="text-xs text-muted-foreground">
                {t('totalQueries')}
              </div>
            </div>
            <div className="text-center p-3 bg-muted/30 rounded-lg">
              <div className="text-2xl font-bold text-accent">
                {sessionStats.totalFeedback > 0 
                  ? `${sessionStats.satisfactionRate.toFixed(0)}%`
                  : (language === 'en' ? 'N/A' : 'Yok')
                }
              </div>
              <div className="text-xs text-muted-foreground">
                {t('satisfaction')}
              </div>
            </div>
          </div>
          
          {sessionStats.totalFeedback > 0 && (
            <div className="text-center">
              <Badge variant="outline" className="text-xs">
                {sessionStats.helpfulCount} / {sessionStats.totalFeedback} {language === 'en' ? 'helpful' : 'yardımcı'}
              </Badge>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-lg">
            ⚡ {language === 'en' ? 'Quick Actions' : 'Hızlı İşlemler'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Button 
            variant="outline" 
            onClick={onClearHistory}
            className="w-full text-destructive hover:text-destructive"
            size="sm"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            {t('clearHistory')}
          </Button>
        </CardContent>
      </Card>

      {/* Help & Info */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-lg">
            ℹ️ {language === 'en' ? 'Information' : 'Bilgi'}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="text-sm space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span>{language === 'en' ? 'Real-time voice input' : 'Gerçek zamanlı ses girişi'}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <span>{language === 'en' ? 'Multi-language support' : 'Çoklu dil desteği'}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
              <span>{language === 'en' ? 'Advanced AI models' : 'Gelişmiş AI modelleri'}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-orange-500 rounded-full"></div>
              <span>{language === 'en' ? 'Airline-specific policies' : 'Havayoluna özel politikalar'}</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};