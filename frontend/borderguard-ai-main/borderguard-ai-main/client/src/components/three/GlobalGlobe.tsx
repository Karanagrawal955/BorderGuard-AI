/*
 * BorderGuard AI visual system: an operational globe rendered as a restrained wireframe intelligence layer.
 * All locations and routes are simulated demonstration data, not live border intelligence.
 */
import { useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Html, OrbitControls, Sparkles } from "@react-three/drei";
import * as THREE from "three";
import { checkpointRoutes, checkpoints, type Checkpoint } from "@/data/checkpoints";

function latLonToVector3(lat: number, lon: number, radius = 2) {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(-radius * Math.sin(phi) * Math.cos(theta), radius * Math.cos(phi), radius * Math.sin(phi) * Math.sin(theta));
}

function arcPoints(a: THREE.Vector3, b: THREE.Vector3) {
  const points: THREE.Vector3[] = [];
  for (let i = 0; i <= 28; i += 1) {
    const t = i / 28;
    const point = new THREE.Vector3().lerpVectors(a, b, t);
    const lift = Math.sin(Math.PI * t) * 0.35;
    point.normalize().multiplyScalar(2.01 + lift);
    points.push(point);
  }
  return points;
}

function FlowParticles({ routes, selectedId }: { routes: { from: string; to: string; points: THREE.Vector3[] }[]; selectedId: string }) {
  const particles = useRef<THREE.Group>(null);
  useFrame((state) => {
    if (!particles.current) return;
    particles.current.children.forEach((child, index) => {
      const route = routes[Math.floor(index / 3)];
      const offset = (index % 3) / 3;
      const progress = (state.clock.elapsedTime * 0.12 + offset) % 1;
      const point = route.points[Math.floor(progress * (route.points.length - 1))];
      child.position.copy(point).multiplyScalar(1.018);
      const isSelectedRoute = route.from === selectedId || route.to === selectedId;
      child.scale.setScalar(isSelectedRoute ? 1.45 : 1);
    });
  });
  return <group ref={particles}>{routes.flatMap((route) => [0, 1, 2].map((particle) => <mesh key={`${route.from}-${route.to}-${particle}`}><sphereGeometry args={[0.035, 8, 8]} /><meshBasicMaterial color={route.from === selectedId || route.to === selectedId ? "#ffffff" : "#42d9ff"} toneMapped={false} /></mesh>))}</group>;
}

function Globe({ selectedId, onSelect }: { selectedId: string; onSelect: (checkpoint: Checkpoint) => void }) {
  const globe = useRef<THREE.Group>(null);
  const points = useMemo(() => new Map(checkpoints.map((checkpoint) => [checkpoint.id, latLonToVector3(checkpoint.lat, checkpoint.lon)])), []);
  const routes = useMemo(() => checkpointRoutes.map(([from, to]) => ({ from, to, points: arcPoints(points.get(from)!, points.get(to)!) })), [points]);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  useFrame((state, delta) => {
    if (globe.current) globe.current.rotation.y += delta * 0.04;
    state.camera.lookAt(0, 0, 0);
  });

  return <group ref={globe} rotation={[0.06, -0.55, 0]}>
    <mesh>
      <sphereGeometry args={[2, 48, 24]} />
      <meshBasicMaterial color="#06212e" transparent opacity={0.85} />
    </mesh>
    <mesh>
      <sphereGeometry args={[2.025, 48, 24]} />
      <meshBasicMaterial color="#42d9ff" wireframe transparent opacity={0.19} toneMapped={false} />
    </mesh>
    <mesh rotation={[0.2, 0.45, 0]}>
      <sphereGeometry args={[2.035, 24, 12]} />
      <meshBasicMaterial color="#b8f5ff" wireframe transparent opacity={0.08} toneMapped={false} />
    </mesh>
    {routes.map(({ points: pointsForRoute, from, to }, index) => <line key={`route-${index}`}>
      <bufferGeometry attach="geometry" setFromPoints={pointsForRoute} />
      <lineBasicMaterial attach="material" color={from === selectedId || to === selectedId ? "#ffffff" : index % 2 === 0 ? "#42d9ff" : "#edb963"} transparent opacity={from === selectedId || to === selectedId ? 0.95 : 0.42} toneMapped={false} />
    </line>)}
    <FlowParticles routes={routes} selectedId={selectedId} />
    {checkpoints.map((checkpoint) => {
      const position = points.get(checkpoint.id)!;
      const selected = selectedId === checkpoint.id;
      const color = checkpoint.status === "CLEAR" ? "#5cf298" : checkpoint.status === "REVIEW" ? "#f1bb66" : "#ff7068";
      return <group key={checkpoint.id} position={position}>
        <mesh onClick={(event) => { event.stopPropagation(); onSelect(checkpoint); }} onPointerOver={(event) => { event.stopPropagation(); setHoveredId(checkpoint.id); document.body.style.cursor = "crosshair"; }} onPointerOut={() => { setHoveredId(null); document.body.style.cursor = ""; }}>
          <sphereGeometry args={[selected ? 0.12 : 0.085, 16, 16]} />
          <meshBasicMaterial color={selected ? "#ffffff" : color} toneMapped={false} />
        </mesh>
        <mesh scale={selected ? 2.3 : 1.7}>
          <sphereGeometry args={[0.085, 16, 16]} />
          <meshBasicMaterial color={color} transparent opacity={0.13} toneMapped={false} />
        </mesh>
        {(selected || hoveredId === checkpoint.id) && <Html distanceFactor={7} position={[0.12, 0.1, 0]}>
          <div className="globe-node-label"><strong>{checkpoint.name}</strong><span>{checkpoint.status} · {checkpoint.id}</span></div>
        </Html>}
      </group>;
    })}
  </group>;
}

export default function GlobalGlobe({ selectedId, onSelect }: { selectedId: string; onSelect: (checkpoint: Checkpoint) => void }) {
  return <div className="global-globe-canvas"><Canvas dpr={[1, 1.5]} camera={{ position: [0, 0.2, 6.3], fov: 37 }} gl={{ antialias: true, powerPreference: "high-performance" }}>
    <color attach="background" args={["#06121a"]} />
    <fog attach="fog" args={["#06121a", 4.5, 11]} />
    <ambientLight intensity={0.7} color="#78bed0" />
    <pointLight position={[2, 3, 5]} intensity={3} color="#42d9ff" />
    <pointLight position={[-3, 1, 2]} intensity={1.6} color="#edb963" />
    <Globe selectedId={selectedId} onSelect={onSelect} />
    <Sparkles count={90} scale={[7, 5, 5]} size={1.2} speed={0.16} color="#42d9ff" opacity={0.45} />
    <OrbitControls enablePan={false} minDistance={4.8} maxDistance={8} rotateSpeed={0.55} zoomSpeed={0.55} />
  </Canvas></div>;
}
