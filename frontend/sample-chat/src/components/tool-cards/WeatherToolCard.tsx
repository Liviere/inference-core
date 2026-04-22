import { useCallback, useEffect, useRef, useState } from 'react';
import type { ToolCallState, ToolMessage } from './utils';
import { tryParseJSON } from './utils';

/**
 * Weather card — inspired by the official LangChain ``tool-calling`` example.
 *
 * WHY the hefty gradient + glow: matches the reference UI so the
 * pattern looks the same in both playgrounds. Colours are defined
 * locally (not via tokens) because the card is meant to be visually
 * decoupled from the rest of the surface palette.
 */

type WeatherArgs = { city?: string } & Record<string, unknown>;

type WeatherTheme = 'sunny' | 'cloudy' | 'rainy';

interface WeatherData {
	city: string;
	temperature: number;
	condition: string;
	unit: string;
	humidity?: number;
	wind_speed?: number;
}

const WEATHER_FONT =
	'"SF Pro Display", "SF Pro Text", Inter, "Noto Sans", system-ui, sans-serif';

const gradientsLight: Record<WeatherTheme, { bg: string; overlay: string }> = {
	sunny: {
		bg: 'linear-gradient(135deg, #38bdf8 0%, #0ea5e9 40%, #2563eb 100%)',
		overlay:
			'radial-gradient(ellipse at 80% 20%, rgba(250,204,21,0.25) 0%, transparent 60%)',
	},
	cloudy: {
		bg: 'linear-gradient(135deg, #94a3b8 0%, #64748b 50%, #475569 100%)',
		overlay:
			'radial-gradient(ellipse at 30% 30%, rgba(203,213,225,0.2) 0%, transparent 60%)',
	},
	rainy: {
		bg: 'linear-gradient(135deg, #64748b 0%, #475569 40%, #334155 100%)',
		overlay:
			'radial-gradient(ellipse at 50% 0%, rgba(148,163,184,0.15) 0%, transparent 50%)',
	},
};

const gradientsDark: Record<WeatherTheme, { bg: string; overlay: string }> = {
	sunny: {
		bg: 'linear-gradient(135deg, #1e3a5f 0%, #0c4a6e 40%, #1e40af 100%)',
		overlay:
			'radial-gradient(ellipse at 80% 20%, rgba(250,204,21,0.12) 0%, transparent 60%)',
	},
	cloudy: {
		bg: 'linear-gradient(135deg, #334155 0%, #1e293b 50%, #0f172a 100%)',
		overlay:
			'radial-gradient(ellipse at 30% 30%, rgba(148,163,184,0.08) 0%, transparent 60%)',
	},
	rainy: {
		bg: 'linear-gradient(135deg, #1e293b 0%, #0f172a 40%, #020617 100%)',
		overlay:
			'radial-gradient(ellipse at 50% 0%, rgba(100,116,139,0.1) 0%, transparent 50%)',
	},
};

function getTheme(condition: string): WeatherTheme {
	const c = condition.toLowerCase();
	if (c.includes('sunny') || c.includes('clear')) return 'sunny';
	if (c.includes('rain') || c.includes('storm') || c.includes('drizzle'))
		return 'rainy';
	return 'cloudy';
}

function useIsDark() {
	const [isDark, setIsDark] = useState(() =>
		typeof document === 'undefined'
			? false
			: document.documentElement.classList.contains('dark')
	);
	useEffect(() => {
		const obs = new MutationObserver(() =>
			setIsDark(document.documentElement.classList.contains('dark'))
		);
		obs.observe(document.documentElement, {
			attributes: true,
			attributeFilter: ['class'],
		});
		return () => obs.disconnect();
	}, []);
	return isDark;
}

function WeatherIcon({
	condition,
	theme,
}: {
	condition: string;
	theme: WeatherTheme;
}) {
	const c = condition.toLowerCase();
	if (c.includes('sunny') || c.includes('clear')) {
		return (
			<svg className="w-12 h-12 drop-shadow-lg" viewBox="0 0 24 24" fill="none">
				<circle cx="12" cy="12" r="4" fill="#fbbf24" opacity={0.9} />
				<circle
					cx="12"
					cy="12"
					r="5.5"
					stroke="#fbbf24"
					strokeWidth={0.6}
					opacity={0.3}
				/>
				<path
					d="M12 3v2.5M12 18.5V21M5.64 5.64l1.77 1.77M16.59 16.59l1.77 1.77M3 12h2.5M18.5 12H21M5.64 18.36l1.77-1.77M16.59 7.41l1.77-1.77"
					stroke="#fcd34d"
					strokeWidth={1.5}
					strokeLinecap="round"
					opacity={0.8}
				/>
			</svg>
		);
	}
	if (c.includes('rain')) {
		return (
			<svg className="w-12 h-12 drop-shadow-lg" viewBox="0 0 24 24" fill="none">
				<path
					d="M4 14.2A4 4 0 0 1 6.2 7a5.5 5.5 0 0 1 10.6-.5A4.5 4.5 0 0 1 18 15H6a4 4 0 0 1-2-.8z"
					fill="rgba(255,255,255,0.25)"
					stroke="rgba(255,255,255,0.4)"
					strokeWidth={0.8}
					strokeLinejoin="round"
				/>
				<path
					d="M8.5 17.5v2M12 17.5v2M15.5 17.5v2"
					stroke="rgba(147,197,253,0.8)"
					strokeWidth={1.5}
					strokeLinecap="round"
				/>
			</svg>
		);
	}
	return (
		<svg className="w-12 h-12 drop-shadow-lg" viewBox="0 0 24 24" fill="none">
			<path
				d="M4 17.2A4 4 0 0 1 6.2 10a5.5 5.5 0 0 1 10.6-.5A4.5 4.5 0 0 1 18 18H6a4 4 0 0 1-2-.8z"
				fill={
					theme === 'cloudy' ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.2)'
				}
				stroke="rgba(255,255,255,0.4)"
				strokeWidth={0.8}
				strokeLinejoin="round"
			/>
		</svg>
	);
}

export function WeatherToolCard({
	call,
	result,
	state,
}: {
	call: { name: string; args: WeatherArgs };
	result?: ToolMessage;
	state: ToolCallState;
}) {
	const cardRef = useRef<HTMLDivElement>(null);
	const [glow, setGlow] = useState({ x: 0, y: 0, intensity: 0 });
	const isDark = useIsDark();

	const onMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
		const card = cardRef.current;
		if (!card) return;
		const rect = card.getBoundingClientRect();
		setGlow({
			x: Math.max(0, Math.min(e.clientX - rect.left, rect.width)),
			y: Math.max(0, Math.min(e.clientY - rect.top, rect.height)),
			intensity: 1,
		});
	}, []);
	const onLeave = useCallback(
		() => setGlow((p) => ({ ...p, intensity: 0 })),
		[]
	);

	if (state === 'pending') {
		return (
			<div
				className="relative overflow-hidden rounded-2xl p-5"
				style={{
					background: isDark
						? 'linear-gradient(135deg, #334155 0%, #1e293b 100%)'
						: 'linear-gradient(135deg, #94a3b8 0%, #64748b 100%)',
					fontFamily: WEATHER_FONT,
				}}
			>
				<div className="flex items-center gap-3">
					<div className="w-10 h-10 rounded-full bg-white/10 animate-pulse" />
					<div className="flex-1 space-y-2">
						<div
							className="text-sm font-medium text-white/90"
							style={{ textShadow: '0 1px 4px rgba(0,0,0,0.2)' }}
						>
							Checking weather in {call.args.city ?? '…'}…
						</div>
						<div className="flex gap-2">
							<div className="h-2 w-16 rounded-full bg-white/15 animate-pulse" />
							<div
								className="h-2 w-10 rounded-full bg-white/10 animate-pulse"
								style={{ animationDelay: '0.15s' }}
							/>
						</div>
					</div>
				</div>
			</div>
		);
	}

	const data = tryParseJSON(result?.content) as WeatherData | null;
	if (!data || typeof data !== 'object') return null;

	const theme = getTheme(data.condition);
	const g = isDark ? gradientsDark[theme] : gradientsLight[theme];
	const unit = data.unit === 'fahrenheit' ? 'F' : 'C';
	const hasDetails =
		(data.humidity !== undefined && data.humidity !== null) ||
		(data.wind_speed !== undefined && data.wind_speed !== null);

	return (
		<div
			ref={cardRef}
			className="relative overflow-hidden rounded-2xl select-none"
			style={{ fontFamily: WEATHER_FONT }}
			onMouseMove={onMove}
			onMouseLeave={onLeave}
		>
			<div className="absolute inset-0" style={{ background: g.bg }} />
			<div className="absolute inset-0" style={{ background: g.overlay }} />
			<div
				className="absolute inset-0 pointer-events-none transition-opacity duration-300"
				style={{
					opacity: glow.intensity,
					background: `radial-gradient(circle 120px at ${glow.x}px ${glow.y}px, rgba(255,255,255,${
						isDark ? 0.12 : 0.2
					}), transparent)`,
				}}
			/>
			<div className="relative p-5">
				<div className="flex items-start justify-between">
					<div>
						<h3
							className="text-[11px] font-semibold uppercase tracking-widest text-white/70 mb-1"
							style={{ textShadow: '0 1px 3px rgba(0,0,0,0.15)' }}
						>
							{data.city}
						</h3>
						<div className="flex items-start">
							<span
								className="font-[250] tabular-nums leading-none tracking-tight text-white/95"
								style={{
									fontSize: '48px',
									textShadow: '0 2px 16px rgba(0,0,0,0.3)',
								}}
							>
								{data.temperature}
							</span>
							<span
								className="mt-1 ml-0.5 font-light tabular-nums text-white/60"
								style={{ fontSize: '18px' }}
							>
								°{unit}
							</span>
						</div>
						<div
							className="text-[13px] mt-1 capitalize text-white/75 font-medium tracking-wide"
							style={{ textShadow: '0 1px 3px rgba(0,0,0,0.1)' }}
						>
							{data.condition}
						</div>
					</div>
					<div className="mt-1">
						<WeatherIcon condition={data.condition} theme={theme} />
					</div>
				</div>

				{hasDetails && (
					<div
						className="flex gap-4 mt-4 rounded-xl px-3 py-2.5"
						style={{
							backgroundColor: 'rgba(255,255,255,0.08)',
							backdropFilter: 'blur(12px) saturate(1.2)',
						}}
					>
						{data.humidity !== undefined && data.humidity !== null && (
							<div className="flex items-center gap-1.5">
								<span className="text-[11px] text-white/50 font-medium">
									Humidity
								</span>
								<span className="text-[11px] text-white/90 font-semibold tabular-nums">
									{data.humidity}%
								</span>
							</div>
						)}
						{data.wind_speed !== undefined && data.wind_speed !== null && (
							<div className="flex items-center gap-1.5">
								<span className="text-[11px] text-white/50 font-medium">
									Wind
								</span>
								<span className="text-[11px] text-white/90 font-semibold tabular-nums">
									{data.wind_speed} mph
								</span>
							</div>
						)}
					</div>
				)}
			</div>
		</div>
	);
}
