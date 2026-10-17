
"""Reusable, import-safe SentinelFlow analytics."""
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from core import COLUMNS, ZONES, overall


def prepare_data(df):
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError('Expected a pandas DataFrame')
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError('Missing CSV columns: ' + ', '.join(sorted(missing)))
    out = df[COLUMNS].copy()
    out['Frame'] = pd.to_numeric(out['Frame'], errors='coerce')
    out['Count'] = pd.to_numeric(out['Count'], errors='coerce')
    out['Density'] = pd.to_numeric(out['Density'], errors='coerce')
    out['Timestamp'] = pd.to_datetime(out['Timestamp'], errors='coerce', utc=True)
    out['Status'] = out['Status'].astype(str).str.upper()
    out = out.dropna(subset=['Frame','Count','Density','Timestamp'])
    out = out[(out.Frame >= 0) & (out.Count >= 0) & (out.Density >= 0) &
              out.Status.isin(['SAFE','MODERATE','DANGER']) & out.Zone.isin(ZONES)]
    if out.duplicated(['Frame','Zone']).any():
        raise ValueError('Duplicate frame/zone records; use one processing run per dataset')
    return out.sort_values(['Frame','Zone']).reset_index(drop=True)


def latest_rows(df):
    return df[df.Frame == df.Frame.max()] if not df.empty else df


def compute_zone_stats(df):
    stats = {}
    for zone in ZONES:
        subset = df[df.Zone == zone].sort_values('Frame')
        if subset.empty:
            continue
        peak = subset.loc[subset.Count.idxmax()]
        dangers = int(subset.Status.eq('DANGER').sum())
        stats[zone] = dict(average_count=round(float(subset.Count.mean()),2),
                           max_count=int(subset.Count.max()), latest_count=int(subset.Count.iloc[-1]),
                           danger_instances=dangers, danger_percent=round(100*dangers/len(subset),1),
                           peak_frame=int(peak.Frame), peak_time=peak.Timestamp.isoformat())
    return stats


def generate_alerts(df):
    if df.empty:
        return ['No detection records available.']
    alerts = []
    for _, r in latest_rows(df).iterrows():
        if r.Status == 'DANGER':
            alerts.append(f'CURRENT DANGER: {r.Zone} has {int(r.Count)} detections.')
        elif r.Status == 'MODERATE':
            alerts.append(f'CURRENT CAUTION: {r.Zone} has {int(r.Count)} detections.')
    for zone, s in compute_zone_stats(df).items():
        if s['danger_percent'] >= 50:
            alerts.append(f'PERSISTENT RISK (prototype): {zone} DANGER in {s["danger_percent"]}% of sampled frames.')
    # Compare each zone with its own baseline, not with other zones.
    for zone in ZONES:
        subset = df[df.Zone == zone]
        if len(subset) < 10:
            continue
        mu, sigma = subset.Count.mean(), subset.Count.std()
        if pd.notna(sigma) and sigma > 0:
            spikes = int((subset.Count > mu + 2*sigma).sum())
            if spikes:
                alerts.append(f'STATISTICAL ANOMALY: {zone} has {spikes} unusually high zone-frame observations (not proof of an emergency).')
    return alerts or ['No current caution/danger or historical warning found.']


def generate_visualizations(df):
    figures = {}
    for key, column, title, ylabel in [('counts','Count','Detected people by frame','People detected'),
                                        ('density','Density','Density index by frame','Detections per pixel')]:
        fig, ax = plt.subplots(figsize=(9,3.5))
        for zone in ZONES:
            subset = df[df.Zone == zone].sort_values('Frame')
            if not subset.empty:
                ax.plot(subset.Frame, subset[column], label=zone)
        ax.set(title=title, xlabel='Processed frame', ylabel=ylabel)
        ax.legend(loc='best'); ax.grid(alpha=.2); fig.tight_layout()
        figures[key] = fig
    fig, ax = plt.subplots(figsize=(9,3.5))
    means = df.groupby('Zone').Count.mean().reindex(ZONES).fillna(0)
    ax.bar(means.index, means.values)
    ax.set(title='Average crowd distribution', ylabel='Mean detected people per zone-frame')
    fig.tight_layout(); figures['distribution'] = fig
    return figures


def generate_rule_based_report(df):
    if df.empty:
        return 'No valid detection data.'
    totals = df.groupby('Frame').Count.sum().sort_index()
    first, last = float(totals.iloc[0]), float(totals.iloc[-1])
    direction = 'increased' if last > first else 'decreased' if last < first else 'remained unchanged'
    return (f'Detected total {direction} from {int(first)} to {int(last)} between the first and last processed frames. '
            f'Peak total was {int(totals.max())} at frame {int(totals.idxmax())}. '
            'Counts represent per-frame detections, not distinct individuals or validated occupancy limits.')


def generate_llm_report(stats):
    import os
    from openai import OpenAI
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('No OPENAI_API_KEY configured')
    client = OpenAI(api_key=key, timeout=15, max_retries=0)
    response = client.chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4o'),
        messages=[{'role':'system','content':'Summarize supplied prototype detection metrics only; do not invent facts or claim real-world safety.'},
                  {'role':'user','content':f'Summarize these measured zone statistics concisely: {stats}'}], temperature=.2)
    return response.choices[0].message.content or ''


def safe_generate_llm_report(stats, df, use_ai=False):
    if use_ai:
        try:
            return 'Optional AI interpretation (not a measured statistic):\n' + generate_llm_report(stats)
        except Exception:
            pass
    return generate_rule_based_report(df)


def generate_report(df, use_ai=False):
    if df.empty:
        return 'SENTINELFLOW REPORT\nNo valid data to report.'
    stats = compute_zone_stats(df)
    latest = latest_rows(df)
    totals = df.groupby('Frame').Count.sum()
    state = overall(latest.Status.tolist())
    lines = ['SENTINELFLOW | CROWD MONITORING REPORT',
             f'Generated: {datetime.now().astimezone().isoformat(timespec="seconds")}',
             f'Frames processed: {df.Frame.nunique()}',
             f'Mean zone-frame count: {df.Count.mean():.2f} (average detected count per zone per processed frame)',
             f'Peak total in one frame: {int(totals.max())}',
             f'Latest total: {int(latest.Count.sum())}', f'Current overall risk: {state}',
             '', 'ZONE STATISTICS']
    for name, s in stats.items():
        lines.append(f'{name}: latest={s["latest_count"]}, average={s["average_count"]}, '
                     f'peak={s["max_count"]} (frame {s["peak_frame"]}), '
                     f'DANGER={s["danger_instances"]} ({s["danger_percent"]}%)')
    lines += ['', 'LATEST ZONE STATUS']
    lines += [f'{r.Zone}: {r.Status} ({int(r.Count)} detected)' for _,r in latest.iterrows()]
    lines += ['', 'CURRENT AND HISTORICAL ALERTS'] + generate_alerts(df)
    lines += ['', 'EXECUTIVE SUMMARY', safe_generate_llm_report(stats, df, use_ai), '',
              'LIMITATIONS', 'These configurable count thresholds are prototype demonstration rules, not validated safety limits.',
              'Density means detected persons divided by zone pixel area, not persons per square meter.',
              'Occlusion, camera perspective, model accuracy and lighting affect detections.',
              'Statistical anomalies are not necessarily emergencies; this system does not predict incidents.']
    return '\n'.join(lines)


if __name__ == '__main__':
    path = Path('crowd_data.csv')
    if not path.is_file():
        raise SystemExit('crowd_data.csv missing. Run detection first.')
    try:
        data = prepare_data(pd.read_csv(path))
        if data.empty:
            raise ValueError('No valid data')
        report = generate_report(data)
        Path('reports.txt').write_text(report, encoding='utf-8')
        print(report)
        for name, fig in generate_visualizations(data).items():
            fig.savefig(f'{name}.png', dpi=140)
            plt.close(fig)
    except (ValueError, pd.errors.EmptyDataError) as exc:
        raise SystemExit(f'Cannot analyze CSV: {exc}')
