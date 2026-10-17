"""Separate public historical staging; never replaces the sealed pilot input.

NOAA GHCN-Daily and EIA history pages remain latest-vintage sources. Their
observation dates and page-level release dates do not certify historical releases.
"""

import argparse
import calendar
import csv
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path

from .cache import Cache
from .providers import fetch_series, _number
from .schema import Observation, assumed_availability
from .validation import validate_observations

ROOT = Path(__file__).resolve().parents[2]
NOAA = 'https://www.ncei.noaa.gov/pub/data/ghcn/daily'
EIA_TABLES = {
    'us_gas_storage': ('nw2_epg0_swo_r48_bcfw', 'Lower 48 States Natural Gas Working Underground Storage', 'Billion Cubic Feet', 'Bcf', 'weekly'),
    'us_gas_consumption': ('n9140us2m', 'U.S. Natural Gas Total Consumption', 'Million Cubic Feet', 'MMcf', 'monthly'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def observation(spec, day, value, metadata, flag='', source=None):
    return Observation(spec['series_id'], day, value, spec['unit'],
        source or f"{spec['provider']}:{spec['source_series']}",
        assumed_availability(day, spec['timezone'], spec['lag_days'], spec['available_hour']),
        metadata['retrieved_at'], 'assumed_lag', 'latest', flag)


def station_metadata(payload, station_id, expected_name):
    # Match the ASCII identifier before decoding names elsewhere in the global
    # inventory, which includes non-ASCII station names.
    matches = [line.decode('utf-8') for line in payload.splitlines() if line[:11] == station_id.encode('ascii')]
    if len(matches) != 1:
        raise ValueError('NOAA station identity missing or duplicated')
    line = matches[0]
    name = line[41:71].strip()
    if name != expected_name:
        raise ValueError('NOAA station name differs from pinned configuration')
    return {'station_id': station_id, 'name': name, 'latitude': float(line[12:20]),
            'longitude': float(line[21:30]), 'elevation_m': float(line[31:37]),
            'state': line[38:40].strip()}


def parse_ghcn(payload, spec, start, end, metadata):
    """Parse documented .dly fixed-width records, retaining flagged/null days.

    Derived temperature is always (TMAX+TMIN)/2, never silently mixed with TAVG.
    The original .dly bytes retain rejected source values and all source flags.
    """
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    station = spec['source_series']
    values = {}
    for line in payload.decode('ascii').splitlines():
        if len(line) != 269 or line[:11] != station:
            raise ValueError('Malformed GHCN record or unexpected station')
        year, month, element = int(line[11:15]), int(line[15:17]), line[17:21]
        month_first = date(year, month, 1)
        month_last = date(year, month, calendar.monthrange(year, month)[1])
        if month_last < first or month_first > last or element not in {'TMAX','TMIN','PRCP'}:
            continue
        for day in range(1, month_last.day+1):
            stamp = date(year, month, day)
            if not first <= stamp <= last:
                continue
            key = (stamp.isoformat(), element)
            if key in values:
                raise ValueError('Duplicate GHCN date/element')
            offset = 21+(day-1)*8
            raw = int(line[offset:offset+5])
            measure, quality, source = line[offset+5:offset+8]
            flag = f'{element}:M={measure.strip()};Q={quality.strip()};S={source.strip()}'
            value = None if raw == -9999 or quality != ' ' else raw/10
            values[key] = (value, flag)
    if not values:
        raise ValueError('No GHCN observations within requested history')
    result = []
    day = first
    while day <= last:
        stamp = day.isoformat()
        tmin, fmin = values.get((stamp,'TMIN'),(None,'TMIN:absent'))
        tmax, fmax = values.get((stamp,'TMAX'),(None,'TMAX:absent'))
        rain, frain = values.get((stamp,'PRCP'),(None,'PRCP:absent'))
        if tmin is not None and tmax is not None and tmin > tmax:
            raise ValueError('GHCN minimum exceeds maximum')
        midpoint = None if tmin is None or tmax is None else (tmin+tmax)/2
        derived = f'{fmin}|{fmax}|method=(TMAX+TMIN)/2'
        fields = {'tmin':(tmin,fmin,'degC'), 'tmax':(tmax,fmax,'degC'),
                  'midrange_temp':(midpoint,derived,'degC'),
                  'hdd18':(None if midpoint is None else max(18-midpoint,0),derived,'degC_day'),
                  'cdd18':(None if midpoint is None else max(midpoint-18,0),derived,'degC_day'),
                  'precip':(rain,frain,'mm')}
        for suffix,(value,flag,unit) in fields.items():
            child = dict(spec, series_id=spec['series_id']+'_'+suffix,unit=unit)
            result.append(observation(child,stamp,value,metadata,flag))
        day += timedelta(days=1)
    return result


class HistoryTable(HTMLParser):
    """Read EIA's labelled historical rows only, without evaluating page scripts."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows=[]; self.row=None; self.cell=None; self.classes=[]; self.title=[]; self.in_title=False

    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if tag=='title': self.in_title=True
        if tag=='tr': self.row=[]; self.classes=[]
        if tag in {'td','th'} and self.row is not None:
            self.cell=[]; self.classes.append(attrs.get('class',''))

    def handle_data(self, data):
        if self.in_title: self.title.append(data)
        if self.cell is not None: self.cell.append(data)

    def handle_endtag(self, tag):
        if tag=='title': self.in_title=False
        if tag in {'td','th'} and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()))
            self.cell=None
        if tag=='tr' and self.row is not None:
            if self.classes and self.classes[0] in {'B4','B6'}:
                self.rows.append((self.row,self.classes))
            self.row=None; self.cell=None


def parse_eia_history(payload, spec, start, end, metadata):
    series = spec['series_id']
    slug,title,unit_text,unit,frequency = EIA_TABLES[series]
    if spec['unit'] != unit or spec['frequency'] != frequency:
        raise ValueError('EIA history unit/frequency mismatch')
    parser=HistoryTable(); parser.feed(payload.decode('utf-8'))
    actual=' '.join(''.join(parser.title).split())
    if actual != f'{title} ({unit_text})':
        raise ValueError('Unexpected EIA history title or units')
    result=[]; seen=set()
    for cells,classes in parser.rows:
        year = int(cells[0][:4])
        if not int(start[:4]) <= year <= int(end[:4]):
            continue
        if frequency=='monthly':
            if len(cells)!=13 or classes != ['B4']+['B3']*12:
                raise ValueError('EIA monthly table layout changed')
            points=[(date(year,m,calendar.monthrange(year,m)[1]).isoformat(),cells[m]) for m in range(1,13)]
        else:
            if len(cells)!=11 or classes != ['B6']+['B5','B3']*5:
                raise ValueError('EIA weekly table layout changed')
            month=datetime.strptime(cells[0],'%Y-%b').month
            points=[]
            for i in range(1,11,2):
                if not cells[i]:
                    if cells[i+1]: raise ValueError('EIA value without date')
                    continue
                stamp=datetime.strptime(f'{year}/{cells[i]}','%Y/%m/%d').date()
                if stamp.month!=month: raise ValueError('EIA weekly date/month mismatch')
                points.append((stamp.isoformat(),cells[i+1]))
        for stamp,raw in points:
            if not start <= stamp <= end: continue
            if stamp in seen: raise ValueError('Duplicate EIA history observation')
            seen.add(stamp)
            value=None if raw in {'','-','--','NA','W'} else _number(raw.replace(',',''))
            result.append(observation(spec,stamp,value,metadata,'source_missing='+raw if value is None else '',
                source=f'eia_history:{slug}'))
    if not result: raise ValueError('No EIA history rows in requested interval')
    return result


def coverage(rows, specs, start, end):
    grouped={s['series_id']:[] for s in specs}
    for row in rows: grouped[row.series_id].append(row)
    result=[]
    periods=[('all',start,end),('2008','2008-01-01','2008-12-31'),('2014-2016','2014-01-01','2016-12-31')]
    for spec in specs:
        for period,first,last in periods:
            first=max(start,first); last=min(end,last)
            if first>last: continue
            selected=[r for r in grouped[spec['series_id']] if first<=r.observation_date<=last]
            finite=[r for r in selected if r.value is not None]
            days=(date.fromisoformat(last)-date.fromisoformat(first)).days+1
            if spec['frequency']=='daily': expected=days
            elif spec['frequency']=='business_daily':
                expected=sum((date.fromisoformat(first)+timedelta(days=i)).weekday()<5 for i in range(days))
            elif spec['frequency']=='monthly': expected=(int(last[:4])-int(first[:4]))*12+int(last[5:7])-int(first[5:7])+1
            else: expected=(days+6)//7
            result.append({'series_id':spec['series_id'],'period':period,'frequency':spec['frequency'],
                'rows':len(selected),'finite_rows':len(finite),'null_rows':len(selected)-len(finite),
                'expected_periods_approx':expected,'finite_coverage_approx':len(finite)/expected if expected else None,
                'first':min((r.observation_date for r in selected),default=''),
                'last':max((r.observation_date for r in selected),default=''),
                'calendar_basis':'calendar days' if spec['frequency']=='daily' else 'approximate; not official exchange/release sessions'})
    return result


def run_history(config_path=ROOT/'config/history.json', *, output_root=ROOT, offline=False, pinned_records=None, publish=True):
    """Stage observations and diagnostics, including failed quality checks.

    Inspect manifest.status before consuming data. A validation_failed run is
    reproducible evidence of rejected data, not an approved training dataset.
    Acquisition/parser failures raise after saving a failure manifest.
    """
    config_path=Path(config_path); output_root=Path(output_root)
    config=json.loads(config_path.read_text())
    start,end=config['start'],config['end']
    if date.fromisoformat(start)>date.fromisoformat(end) or end>'2025-12-31':
        raise ValueError('Historical extension is restricted to already-exposed history ending by 2025-12-31')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder=output_root/'runs'/f'history-{stamp}'; folder.mkdir(parents=True,exist_ok=False)
    cache=Cache(output_root/'data/extensions/raw',offline=offline,pinned_records=pinned_records)
    manifest={'phase':'historical-extension','status':'started','historical_point_in_time_certified':False,
              'scope':'Additional latest-vintage exploratory staging; no model fitting or new holdout evaluation',
              'artifacts':{},'sources':[]}
    try:
        write_json(folder/'config.json',config)
        consumed=['historical.py','cache.py','providers.py','schema.py','validation.py']
        snapshot={name:(Path(__file__).parent/name).read_text() for name in consumed}
        write_json(folder/'source_snapshot.json',snapshot)
        rows=[]; specs=[]; stations=[]
        for spec in config['market_series']:
            print('Fetching '+spec['series_id'],flush=True)
            rows.extend(fetch_series(spec,start,end,cache)); specs.append(spec)
        station_bytes,_=cache.get(NOAA+'/ghcnd-stations.txt')
        for spec in config['weather_stations']:
            print('Fetching '+spec['series_id'],flush=True)
            stations.append(station_metadata(station_bytes,spec['source_series'],spec['station_name']))
            data,meta=cache.get(NOAA+'/all/'+spec['source_series']+'.dly')
            parsed=parse_ghcn(data,spec,start,end,meta); rows.extend(parsed)
            for row in parsed[:6]:
                specs.append(dict(spec,series_id=row.series_id,unit=row.unit,frequency='daily',max_missing_fraction=0.12,
                    minimum=0 if row.unit in {'mm','degC_day'} else -90,maximum=1000 if row.unit=='mm' else 100))
        for spec in config['eia_history_series']:
            print('Fetching '+spec['series_id'],flush=True)
            slug=EIA_TABLES[spec['series_id']][0]
            data,meta=cache.get(f'https://www.eia.gov/dnav/ng/hist/{slug}.htm')
            rows.extend(parse_eia_history(data,spec,start,end,meta)); specs.append(spec)
        ids=[s['series_id'] for s in specs]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate configured series')
        rows.sort(key=lambda r:(r.series_id,r.observation_date))
        report=validate_observations(rows,specs,start,end)
        write_json(folder/'validation.json',report)
        with (folder/'observations.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(Observation.__dataclass_fields__))
            writer.writeheader(); writer.writerows(asdict(r) for r in rows)
        stats=coverage(rows,specs,start,end)
        with (folder/'coverage.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(stats[0])); writer.writeheader(); writer.writerows(stats)
        write_json(folder/'station_metadata.json',stations)
        manifest.update(status='passed' if report['passed'] else 'validation_failed',
                        observation_rows=len(rows),series=len(specs),
                        code_sha256={name:hashlib.sha256(value.encode()).hexdigest() for name,value in snapshot.items()})
        lines=['# Historical extension coverage','',
            'Separate latest-vintage data staging for '+start+' to '+end+'. No model forecasts or new holdout scores were computed.','',
            '**Data validation: '+('passed.' if report['passed'] else 'FAILED. This snapshot is not approved for model fitting.')+'**','']
        if report['errors']:
            lines += ['- '+error for error in report['errors']] + ['']
        lines += [
            '| Series | Period | Finite rows | Approx. expected | First | Last |',
            '|---|---|---:|---:|---|---|']
        for r in stats:
            lines.append(f"| {r['series_id']} | {r['period']} | {r['finite_rows']} | {r['expected_periods_approx']} | {r['first']} | {r['last']} |")
        lines += ['','NOAA features use (TMAX+TMIN)/2 and are named midrange temperatures; they are not observed daily means or archived forecasts. Nonblank quality flags are withheld and preserved in raw snapshots. Three airport locations are geographic proxies, not population-weighted exposure indices.',
            '', 'FRED DEXCAUS is a separate noon-buying-rate CAD/USD series; it is not spliced into the pilot Bank of Canada daily-average series. EIA history tables require no API key but remain revised history, not release-vintage archives. Missing early storage coverage is explicit.',
            '', 'All availability delays remain assumptions. Coverage denominators for financial/weekly/monthly series are approximate, not executable market calendars. Existing Phase 1-7 results and their sealed input remain unchanged.']
        (folder/'history_report.md').write_text('\n'.join(lines)+'\n')
    except Exception as exc:
        manifest.update(status='failed',failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['sources']=[{k:v for k,v in r.items() if k!='cache_hit'} for r in cache.records]
        manifest['artifacts']={p.name:sha(p) for p in sorted(folder.iterdir()) if p.name!='manifest.json'}
        write_json(folder/'manifest.json',manifest)
    if publish:
        dest=output_root/'reports/extensions'; dest.mkdir(parents=True,exist_ok=True)
        for name in ['history_report.md','coverage.csv','validation.json','station_metadata.json','manifest.json']:
            (dest/name).write_bytes((folder/name).read_bytes())
    return folder


def replay_history(manifest_path, *, output_root=ROOT, publish=False):
    """Verify saved code/artifacts, pin source blobs, and compare every output."""
    manifest_path=Path(manifest_path)
    original=json.loads(manifest_path.read_text())
    if original.get('status') not in {'passed','validation_failed'}:
        raise ValueError('Only completed staging runs can be replayed')
    for name,digest in original['artifacts'].items():
        if Path(name).name != name or sha(manifest_path.parent/name) != digest:
            raise ValueError('Historical run artifact changed: '+name)
    for name,digest in original['code_sha256'].items():
        if Path(name).name != name or sha(Path(__file__).parent/name) != digest:
            raise ValueError('Historical source changed: '+name)
    result=run_history(manifest_path.parent/'config.json',output_root=output_root,
                       offline=True,pinned_records=original['sources'],publish=False)
    if json.loads((result/'manifest.json').read_text()) != original:
        raise ValueError('Historical replay differs; new run preserved at '+str(result))
    if publish:
        dest=Path(output_root)/'reports/extensions'; dest.mkdir(parents=True,exist_ok=True)
        for name in ['history_report.md','coverage.csv','validation.json','station_metadata.json','manifest.json']:
            (dest/name).write_bytes((result/name).read_bytes())
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/history.json')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--no-publish',action='store_true')
    parser.add_argument('--replay',type=Path,help='Replay a completed manifest with pinned source bytes')
    args=parser.parse_args(argv)
    result=(replay_history(args.replay,publish=not args.no_publish) if args.replay else
            run_history(args.config,offline=args.offline,publish=not args.no_publish))
    print('Historical extension staged:',result)
    passed=json.loads((result/'manifest.json').read_text())['status']=='passed'
    if not passed: print('Data validation FAILED; see validation.json. Do not use this snapshot for model fitting.')
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
