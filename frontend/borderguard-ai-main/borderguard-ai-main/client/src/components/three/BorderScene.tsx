/*
 * BorderGuard AI visual system: cinematic R3F scene with restrained navy, cyan emissive light,
 * technical geometry, and a clear demo-only border-security environment.
 */
import { useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, Line, OrbitControls, Sparkles, Text } from "@react-three/drei";
import * as THREE from "three";
import type { ScanState } from "@/hooks/useScanSimulation";

type BorderSceneProps = {
  scanState: ScanState;
  reduced?: boolean;
};

function Scanner({ scanState }: { scanState: ScanState }) {
  const beam = useRef<THREE.Mesh>(null);
  const hologram = useRef<THREE.Group>(null);
  const active = scanState !== "IDLE";
  const warm = scanState === "TAMPERING_ANALYSIS";

  useFrame((state, delta) => {
    if (beam.current) {
      beam.current.position.y = Math.sin(state.clock.elapsedTime * (active ? 2.4 : 0.6)) * 0.58;
      beam.current.scale.x = 1 + Math.sin(state.clock.elapsedTime * 1.9) * 0.04;
    }
    if (hologram.current) {
      hologram.current.rotation.y += delta * (active ? 0.18 : 0.05);
      hologram.current.position.y = 1.5 + Math.sin(state.clock.elapsedTime * 1.3) * 0.04;
    }
  });

  return (
    <group position={[0, -0.3, 0]}>
      <mesh position={[0, -0.35, 0]} castShadow receiveShadow>
        <boxGeometry args={[3.7, 0.45, 2.45]} />
        <meshStandardMaterial color="#0b1820" metalness={0.9} roughness={0.28} />
      </mesh>
      <mesh position={[0, -0.08, 0]} rotation={[-0.08, 0, 0]} castShadow>
        <boxGeometry args={[3.15, 0.16, 1.92]} />
        <meshPhysicalMaterial color="#0e2933" metalness={0.65} roughness={0.18} transmission={0.15} transparent opacity={0.95} />
      </mesh>
      <mesh position={[0, 0.03, 0]} rotation={[-0.08, 0, 0]}>
        <boxGeometry args={[2.82, 0.035, 1.62]} />
        <meshBasicMaterial color={warm ? "#edb963" : "#42d9ff"} toneMapped={false} />
      </mesh>
      <mesh ref={beam} position={[0, 0.55, 0]}>
        <boxGeometry args={[2.7, 0.018, 1.5]} />
        <meshBasicMaterial color="#b7f7ff" transparent opacity={active ? 0.8 : 0.18} toneMapped={false} />
      </mesh>
      <mesh position={[-1.82, -0.1, 0]}>
        <boxGeometry args={[0.04, 0.42, 1.8]} />
        <meshBasicMaterial color="#42d9ff" toneMapped={false} />
      </mesh>
      <mesh position={[1.82, -0.1, 0]}>
        <boxGeometry args={[0.04, 0.42, 1.8]} />
        <meshBasicMaterial color="#42d9ff" toneMapped={false} />
      </mesh>
      <Float speed={active ? 1.2 : 0.4} rotationIntensity={0.08} floatIntensity={0.12}>
        <group ref={hologram} position={[0, 1.5, 0]}>
          <mesh>
            <boxGeometry args={[2.25, 1.28, 0.035]} />
            <meshBasicMaterial color="#42d9ff" wireframe transparent opacity={active ? 0.35 : 0.16} toneMapped={false} />
          </mesh>
          <Line points={[[-1.15, 0.56, 0], [1.15, 0.56, 0], [1.15, -0.56, 0], [-1.15, -0.56, 0], [-1.15, 0.56, 0]]} color={warm ? "#edb963" : "#42d9ff"} lineWidth={1.2} transparent opacity={active ? 0.9 : 0.42} />
          <Text position={[-0.75, 0.31, 0.04]} fontSize={0.105} color="#baf4ff" anchorX="left" anchorY="middle" font="https://fonts.gstatic.com/s/ibmplexmono/v19/-F6jfjtqLzI2JPCgQBnw7HFyzSD-AsregP8VFBEj35zS0w.woff2">
            DOCUMENT SCAN
          </Text>
          <Text position={[-0.75, 0.1, 0.04]} fontSize={0.08} color="#6eb1c2" anchorX="left" anchorY="middle" font="https://fonts.gstatic.com/s/ibmplexmono/v19/-F6jfjtqLzI2JPCgQBnw7HFyzSD-AsregP8VFBEj35zS0w.woff2">
            OCR  ·  FACE  ·  MRZ  ·  SECURE
          </Text>
          <mesh position={[0.65, -0.12, 0.04]}>
            <circleGeometry args={[0.26, 24]} />
            <meshBasicMaterial color="#42d9ff" wireframe transparent opacity={0.62} toneMapped={false} />
          </mesh>
        </group>
      </Float>
      <mesh position={[0, 0.23, -0.76]} rotation={[0, 0, 0]}>
        <boxGeometry args={[1.9, 0.035, 0.03]} />
        <meshBasicMaterial color="#edb963" toneMapped={false} />
      </mesh>
    </group>
  );
}

function WorldGrid({ scanState }: { scanState: ScanState }) {
  const group = useRef<THREE.Group>(null);
  useFrame((_, delta) => {
    if (group.current) group.current.rotation.y += delta * 0.025;
  });
  return (
    <group ref={group} position={[0, 1.1, -3.4]} rotation={[0.15, 0, 0]}>
      <mesh>
        <sphereGeometry args={[1.65, 32, 18]} />
        <meshBasicMaterial color="#0a2736" wireframe transparent opacity={scanState === "RISK_ANALYSIS" ? 0.45 : 0.2} />
      </mesh>
      <mesh scale={[1.04, 1.04, 1.04]}>
        <sphereGeometry args={[1.65, 32, 18]} />
        <meshBasicMaterial color="#42d9ff" wireframe transparent opacity={0.055} toneMapped={false} />
      </mesh>
      <Line points={[[-1.4, 0.3, 0], [0.3, 0.8, 0], [1.25, -0.3, 0]]} color="#42d9ff" lineWidth={1} transparent opacity={0.62} />
      <Line points={[[0.3, 0.8, 0], [0.95, 1.15, 0]]} color="#edb963" lineWidth={1} transparent opacity={0.62} />
      <mesh position={[0.3, 0.8, 0.02]}>
        <sphereGeometry args={[0.07, 12, 12]} />
        <meshBasicMaterial color="#edb963" toneMapped={false} />
      </mesh>
    </group>
  );
}

function CameraRig({ scanState }: { scanState: ScanState }) {
  useFrame((state) => {
    const target = scanState === "IDLE" ? 5.9 : scanState === "VERIFIED" ? 5.5 : 5.1;
    state.camera.position.z = THREE.MathUtils.lerp(state.camera.position.z, target, 0.025);
    state.camera.position.x = THREE.MathUtils.lerp(state.camera.position.x, state.pointer.x * 0.18, 0.03);
    state.camera.position.y = THREE.MathUtils.lerp(state.camera.position.y, 1.7 + state.pointer.y * 0.1, 0.03);
    state.camera.lookAt(0, 0.45, 0);
  });
  return null;
}

export default function BorderScene({ scanState, reduced = false }: BorderSceneProps) {
  return (
    <Canvas dpr={[1, 1.55]} camera={{ position: [0, 1.7, 5.9], fov: 38 }} gl={{ antialias: true, powerPreference: "high-performance" }}>
      <color attach="background" args={["#06111a"]} />
      <fog attach="fog" args={["#06111a", 4, 12]} />
      <ambientLight intensity={0.35} color="#83b9c8" />
      <pointLight position={[0, 3, 2]} intensity={3.2} distance={7} color="#42d9ff" />
      <pointLight position={[-3, 0, 1]} intensity={1.4} distance={5} color="#edb963" />
      <Scanner scanState={scanState} />
      {!reduced && <WorldGrid scanState={scanState} />}
      {!reduced && <Sparkles count={65} scale={[7, 4, 4]} size={1.5} speed={0.22} color="#42d9ff" opacity={0.45} />}
      <CameraRig scanState={scanState} />
      <OrbitControls enablePan={false} enableZoom={false} enableRotate={!reduced} minPolarAngle={1.1} maxPolarAngle={1.95} />
    </Canvas>
  );
}
