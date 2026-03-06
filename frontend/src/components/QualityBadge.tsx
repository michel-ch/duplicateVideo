

interface Props {
  score: number | null;
  showBar?: boolean;
}

export default function QualityBadge({ score, showBar = true }: Props) {
  if (score === null || score === undefined) return <span className="badge badge-muted">N/A</span>;

  const level = score >= 70 ? 'high' : score >= 40 ? 'medium' : 'low';
  const colorClass = level === 'high' ? 'green' : level === 'medium' ? 'yellow' : 'red';

  return (
    <div className="quality-gauge">
      {showBar && (
        <div className="quality-bar-bg">
          <div
            className={`quality-bar-fill ${level}`}
            style={{ width: `${Math.min(score, 100)}%` }}
          />
        </div>
      )}
      <span className={`quality-value text-${colorClass}`}>
        {score.toFixed(1)}
      </span>
    </div>
  );
}
