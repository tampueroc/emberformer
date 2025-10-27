# Coordinate Transform & DUT Cross-Link Implementation Plan

## Executive Summary

This document outlines the implementation plan for adding per-session coordinate transforms and DUT cross-linking capabilities to the production data system. The solution will enable normalization of device coordinates across different wafer orientations and maintain lookup links for retrieving all related data for a given Device Under Test (DUT).

**Target Repositories:**
- `SA_15_production_data_storage` - Database schema and data access layer
- `SA-21_production_data_ingester` - STDF file ingestion pipeline

**Tech Stack Identified:**
- **Language:** Python 3.x
- **Database:** MySQL
- **ORM:** SQLAlchemy with Alembic migrations
- **Existing Models:** `TestSession`, `WaferLevelDie`, `STDFParametricData`, `PartSummary`
- **Deployment:** AWS Lambda (ingester), Serverless Framework

---

## Phase 1: Schema Design & Migrations

### 1.1 New Database Tables

#### Table: `orientation_transform`
Stores coordinate transformation metadata per test session.

**Fields:**
```python
- id: Integer (PK)
- test_session_id: Integer (FK → test_session.id, UNIQUE)
- rotation_degrees: Float (0-360, default: 0.0)
- translation_x: Float (default: 0.0)
- translation_y: Float (default: 0.0)
- flip_x: Boolean (default: False)
- flip_y: Boolean (default: False)
- scale: Float (default: 1.0)
- coordinate_frame: String(50) (e.g., "stdf_raw", "normalized_wafer")
- version: Integer (default: 1, for tracking corrections)
- is_active: Boolean (default: True, for soft deletes)
- source: Enum["manual", "auto_header", "prober_recipe", "config"] (provenance)
- confidence: Enum["high", "medium", "low", "unknown"]
- notes: String(500) (nullable, for human context)
- created_at: DateTime (UTC)
- updated_at: DateTime (UTC)
- created_by: String(30)
- updated_by: String(30)
```

**Indexes:**
- Primary key on `id`
- Unique index on `(test_session_id, version)` when `is_active=True`
- Index on `test_session_id` for fast lookups

**Constraints:**
- `rotation_degrees` between 0 and 360
- `scale` > 0
- Foreign key to `test_session.id` with ON DELETE CASCADE

---

#### Table: `dut_link`
Maps canonical DUT identifiers to source-specific identifiers across sessions.

**Fields:**
```python
- id: Integer (PK)
- dut_uid: String(64) (indexed, canonical identifier)
- test_session_id: Integer (FK → test_session.id)
- source_type: Enum["stdf_parametric", "part_summary", "wafer_die", "image", "log"]
- source_record_id: Integer (nullable, points to specific record)
- die_x: Integer (nullable, raw STDF coordinate)
- die_y: Integer (nullable, raw STDF coordinate)
- normalized_x: Float (nullable, transformed coordinate)
- normalized_y: Float (nullable, transformed coordinate)
- site_num: Integer (nullable)
- stdf_specific_dut_serial: String(32) (nullable)
- wafer_id: String(20) (nullable)
- supplier: String(60) (denormalized for fast filtering)
- product: String(60) (denormalized for fast filtering)
- lot_id: String(60) (denormalized for fast filtering)
- created_at: DateTime (UTC)
- updated_at: DateTime (UTC)
- created_by: String(30)
- updated_by: String(30)
```

**Indexes:**
- Primary key on `id`
- Index on `dut_uid` (high cardinality, frequent lookups)
- Composite index on `(supplier, product, lot_id, wafer_id, normalized_x, normalized_y)`
- Index on `test_session_id`
- Index on `(source_type, source_record_id)`

**Constraints:**
- Unique constraint on `(dut_uid, test_session_id, source_type, source_record_id)`
- Check constraint: at least one of (`source_record_id`, `die_x`/`die_y`, `stdf_specific_dut_serial`) must be non-null

---

#### Table: `normalized_coordinates_cache` (Optional Materialized View)
Pre-computed normalized coordinates for fast queries without on-the-fly transforms.

**Fields:**
```python
- id: Integer (PK)
- stdf_parametric_data_id: Integer (FK → stdf_parametric_data.id, UNIQUE)
- test_session_id: Integer (FK → test_session.id, indexed)
- dut_uid: String(64) (indexed)
- raw_x: Integer
- raw_y: Integer
- normalized_x: Float
- normalized_y: Float
- transform_version: Integer (FK → orientation_transform.version)
- is_stale: Boolean (default: False, flagged when transform updates)
- created_at: DateTime (UTC)
- updated_at: DateTime (UTC)
```

**Indexes:**
- Primary key on `id`
- Unique index on `stdf_parametric_data_id`
- Composite index on `(test_session_id, dut_uid)`
- Index on `is_stale` for cache refresh queries

---

### 1.2 Migration Files

#### Migration 1: `add_orientation_transform_table.py`

**Up Migration:**
```python
def upgrade() -> None:
    op.create_table(
        'orientation_transform',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('test_session_id', sa.Integer(), nullable=False),
        sa.Column('rotation_degrees', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('translation_x', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('translation_y', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('flip_x', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('flip_y', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('scale', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('coordinate_frame', sa.String(50), nullable=False, server_default='stdf_raw'),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('source', sa.Enum('manual', 'auto_header', 'prober_recipe', 'config'), nullable=False),
        sa.Column('confidence', sa.Enum('high', 'medium', 'low', 'unknown'), nullable=False, server_default='unknown'),
        sa.Column('notes', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_by', sa.String(30), nullable=False),
        sa.Column('updated_by', sa.String(30), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['test_session_id'], ['test_session.id'], ondelete='CASCADE'),
        sa.CheckConstraint('rotation_degrees >= 0 AND rotation_degrees <= 360', name='check_rotation_range'),
        sa.CheckConstraint('scale > 0', name='check_scale_positive')
    )
    op.create_index('idx_orientation_test_session', 'orientation_transform', ['test_session_id'])
    op.create_index('idx_orientation_active', 'orientation_transform', ['test_session_id', 'version', 'is_active'], unique=True)
```

**Down Migration:**
```python
def downgrade() -> None:
    op.drop_index('idx_orientation_active', table_name='orientation_transform')
    op.drop_index('idx_orientation_test_session', table_name='orientation_transform')
    op.drop_table('orientation_transform')
```

---

#### Migration 2: `add_dut_link_table.py`

**Up Migration:**
```python
def upgrade() -> None:
    op.create_table(
        'dut_link',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('dut_uid', sa.String(64), nullable=False),
        sa.Column('test_session_id', sa.Integer(), nullable=False),
        sa.Column('source_type', sa.Enum('stdf_parametric', 'part_summary', 'wafer_die', 'image', 'log'), nullable=False),
        sa.Column('source_record_id', sa.Integer(), nullable=True),
        sa.Column('die_x', sa.Integer(), nullable=True),
        sa.Column('die_y', sa.Integer(), nullable=True),
        sa.Column('normalized_x', sa.Float(), nullable=True),
        sa.Column('normalized_y', sa.Float(), nullable=True),
        sa.Column('site_num', sa.Integer(), nullable=True),
        sa.Column('stdf_specific_dut_serial', sa.String(32), nullable=True),
        sa.Column('wafer_id', sa.String(20), nullable=True),
        sa.Column('supplier', sa.String(60), nullable=True),
        sa.Column('product', sa.String(60), nullable=True),
        sa.Column('lot_id', sa.String(60), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_by', sa.String(30), nullable=False),
        sa.Column('updated_by', sa.String(30), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['test_session_id'], ['test_session.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('dut_uid', 'test_session_id', 'source_type', 'source_record_id', name='uq_dut_link_source')
    )
    op.create_index('idx_dut_link_uid', 'dut_link', ['dut_uid'])
    op.create_index('idx_dut_link_session', 'dut_link', ['test_session_id'])
    op.create_index('idx_dut_link_source', 'dut_link', ['source_type', 'source_record_id'])
    op.create_index('idx_dut_link_location', 'dut_link', ['supplier', 'product', 'lot_id', 'wafer_id', 'normalized_x', 'normalized_y'])
```

**Down Migration:**
```python
def downgrade() -> None:
    op.drop_index('idx_dut_link_location', table_name='dut_link')
    op.drop_index('idx_dut_link_source', table_name='dut_link')
    op.drop_index('idx_dut_link_session', table_name='dut_link')
    op.drop_index('idx_dut_link_uid', table_name='dut_link')
    op.drop_table('dut_link')
```

---

#### Migration 3: `add_normalized_coordinates_cache.py` (Optional)

**Up/Down migrations** similar to above, creating the cache table with proper indexes.

---

### 1.3 SQLAlchemy Model Updates

**File:** `SA_15_production_data_storage/src/model.py`

Add new model classes following existing conventions:

```python
class TransformSource(StdLibEnum):
    manual = "manual"
    auto_header = "auto_header"
    prober_recipe = "prober_recipe"
    config = "config"

class TransformConfidence(StdLibEnum):
    high = "high"
    medium = "medium"
    low = "low"
    unknown = "unknown"

class SourceType(StdLibEnum):
    stdf_parametric = "stdf_parametric"
    part_summary = "part_summary"
    wafer_die = "wafer_die"
    image = "image"
    log = "log"

class OrientationTransform(Base, TimeStampMixin, AgentMixin):
    """Coordinate transformation metadata per test session"""

    __tablename__ = "orientation_transform"

    id: Mapped[int] = mapped_column(primary_key=True)
    test_session_id: Mapped[int] = mapped_column(
        ForeignKey("test_session.id", ondelete="CASCADE"), nullable=False
    )
    test_session = relationship("TestSession", back_populates="orientation_transform")
    rotation_degrees: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    translation_x: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    translation_y: Mapped[float] = mapped_column(Float(), nullable=False, default=0.0)
    flip_x: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)
    flip_y: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)
    scale: Mapped[float] = mapped_column(Float(), nullable=False, default=1.0)
    coordinate_frame: Mapped[str] = mapped_column(String(50), nullable=False, default="stdf_raw")
    version: Mapped[int] = mapped_column(Integer(), nullable=False, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=True)
    source: Mapped[TransformSource] = mapped_column(Enum(TransformSource), nullable=False)
    confidence: Mapped[TransformConfidence] = mapped_column(Enum(TransformConfidence), nullable=False, default=TransformConfidence.unknown)
    notes: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

class DUTLink(Base, TimeStampMixin, AgentMixin):
    """Cross-reference table linking canonical DUT identifiers to source records"""

    __tablename__ = "dut_link"
    __table_args__ = (
        UniqueConstraint("dut_uid", "test_session_id", "source_type", "source_record_id", name="uq_dut_link_source"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    dut_uid: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    test_session_id: Mapped[int] = mapped_column(
        ForeignKey("test_session.id", ondelete="CASCADE"), nullable=False, index=True
    )
    test_session = relationship("TestSession", back_populates="dut_links")
    source_type: Mapped[SourceType] = mapped_column(Enum(SourceType), nullable=False, index=True)
    source_record_id: Mapped[Optional[int]] = mapped_column(Integer(), nullable=True)
    die_x: Mapped[Optional[int]] = mapped_column(Integer(), nullable=True)
    die_y: Mapped[Optional[int]] = mapped_column(Integer(), nullable=True)
    normalized_x: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    normalized_y: Mapped[Optional[float]] = mapped_column(Float(), nullable=True)
    site_num: Mapped[Optional[int]] = mapped_column(Integer(), nullable=True)
    stdf_specific_dut_serial: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    wafer_id: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    supplier: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
    product: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
    lot_id: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)

# Add relationships to existing TestSession model
# TestSession.orientation_transform = relationship("OrientationTransform", back_populates="test_session", uselist=False)
# TestSession.dut_links = relationship("DUTLink", back_populates="test_session")
```

---

## Phase 2: Data Access Layer (DAL)

### 2.1 Transform Service Module

**New File:** `SA_15_production_data_storage/src/transform_service.py`

```python
"""Service layer for coordinate transformations and DUT linking"""

import hashlib
import math
from typing import Optional, Tuple, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from .model import (
    OrientationTransform,
    DUTLink,
    TestSession,
    TransformSource,
    TransformConfidence,
    SourceType,
    STDFParametricData,
    WaferLevelDie
)

# Coordinate frame definitions
COORDINATE_FRAME_STDF_RAW = "stdf_raw"
COORDINATE_FRAME_NORMALIZED = "normalized_wafer"

# Transform operation order: scale → flip → rotate → translate
# Rotation convention: positive degrees = counter-clockwise (CCW)

class TransformService:
    """Handles coordinate transformations and normalization"""

    def __init__(self, session: Session):
        self.session = session

    def get_or_create_identity_transform(
        self,
        test_session_id: int,
        source: TransformSource = TransformSource.auto_header,
        notes: Optional[str] = None
    ) -> OrientationTransform:
        """Get existing transform or create identity transform for session"""
        existing = self.session.query(OrientationTransform).filter(
            and_(
                OrientationTransform.test_session_id == test_session_id,
                OrientationTransform.is_active == True
            )
        ).order_by(OrientationTransform.version.desc()).first()

        if existing:
            return existing

        identity = OrientationTransform(
            test_session_id=test_session_id,
            rotation_degrees=0.0,
            translation_x=0.0,
            translation_y=0.0,
            flip_x=False,
            flip_y=False,
            scale=1.0,
            source=source,
            confidence=TransformConfidence.unknown,
            notes=notes or "Identity transform (default)",
            created_by="system",
            updated_by="system"
        )
        self.session.add(identity)
        self.session.flush()
        return identity

    def upsert_transform(
        self,
        test_session_id: int,
        rotation_degrees: float = 0.0,
        translation_x: float = 0.0,
        translation_y: float = 0.0,
        flip_x: bool = False,
        flip_y: bool = False,
        scale: float = 1.0,
        source: TransformSource = TransformSource.manual,
        confidence: TransformConfidence = TransformConfidence.medium,
        notes: Optional[str] = None,
        user: str = "system"
    ) -> OrientationTransform:
        """Upsert transform, versioning existing if present"""

        # Validate inputs
        if not (0 <= rotation_degrees <= 360):
            raise ValueError("rotation_degrees must be between 0 and 360")
        if scale <= 0:
            raise ValueError("scale must be positive")

        # Deactivate existing transforms for this session
        existing_transforms = self.session.query(OrientationTransform).filter(
            and_(
                OrientationTransform.test_session_id == test_session_id,
                OrientationTransform.is_active == True
            )
        ).all()

        next_version = 1
        if existing_transforms:
            for t in existing_transforms:
                t.is_active = False
                t.updated_by = user
                next_version = max(next_version, t.version + 1)

        # Create new transform
        new_transform = OrientationTransform(
            test_session_id=test_session_id,
            rotation_degrees=rotation_degrees,
            translation_x=translation_x,
            translation_y=translation_y,
            flip_x=flip_x,
            flip_y=flip_y,
            scale=scale,
            version=next_version,
            source=source,
            confidence=confidence,
            notes=notes,
            created_by=user,
            updated_by=user
        )
        self.session.add(new_transform)
        self.session.flush()

        # TODO: Mark normalized_coordinates_cache as stale for this session

        return new_transform

    def apply_transform(
        self,
        x: float,
        y: float,
        transform: OrientationTransform
    ) -> Tuple[float, float]:
        """Apply transformation to raw coordinates

        Order of operations:
        1. Scale
        2. Flip X/Y
        3. Rotate (CCW positive)
        4. Translate
        """
        # 1. Scale
        x_scaled = x * transform.scale
        y_scaled = y * transform.scale

        # 2. Flip
        x_flipped = -x_scaled if transform.flip_x else x_scaled
        y_flipped = -y_scaled if transform.flip_y else y_scaled

        # 3. Rotate (convert degrees to radians, CCW positive)
        theta = math.radians(transform.rotation_degrees)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        x_rotated = x_flipped * cos_theta - y_flipped * sin_theta
        y_rotated = x_flipped * sin_theta + y_flipped * cos_theta

        # 4. Translate
        x_final = x_rotated + transform.translation_x
        y_final = y_rotated + transform.translation_y

        return (x_final, y_final)

    def get_normalized_coordinates(
        self,
        test_session_id: int,
        raw_x: int,
        raw_y: int
    ) -> Tuple[float, float]:
        """Get normalized coordinates for a given raw position"""
        transform = self.get_or_create_identity_transform(test_session_id)
        return self.apply_transform(raw_x, raw_y, transform)


class DUTLinkService:
    """Handles DUT cross-linking across sources"""

    def __init__(self, session: Session):
        self.session = session

    def generate_dut_uid(
        self,
        supplier: str,
        product: str,
        lot_id: str,
        wafer_id: str,
        normalized_x: float,
        normalized_y: float,
        precision: int = 2
    ) -> str:
        """Generate deterministic canonical DUT UID

        Uses SHA256 hash of normalized coordinates and identifiers.
        Precision controls coordinate rounding for determinism.
        """
        # Round coordinates to specified precision
        x_rounded = round(normalized_x, precision)
        y_rounded = round(normalized_y, precision)

        # Create deterministic string
        components = [
            supplier.strip().upper(),
            product.strip().upper(),
            lot_id.strip().upper(),
            wafer_id.strip().upper(),
            f"{x_rounded:.{precision}f}",
            f"{y_rounded:.{precision}f}"
        ]
        uid_string = "|".join(components)

        # Hash to fixed-length identifier
        hash_obj = hashlib.sha256(uid_string.encode('utf-8'))
        return hash_obj.hexdigest()[:64]

    def create_dut_link(
        self,
        dut_uid: str,
        test_session_id: int,
        source_type: SourceType,
        source_record_id: Optional[int] = None,
        die_x: Optional[int] = None,
        die_y: Optional[int] = None,
        normalized_x: Optional[float] = None,
        normalized_y: Optional[float] = None,
        site_num: Optional[int] = None,
        stdf_specific_dut_serial: Optional[str] = None,
        wafer_id: Optional[str] = None,
        supplier: Optional[str] = None,
        product: Optional[str] = None,
        lot_id: Optional[str] = None,
        user: str = "system"
    ) -> DUTLink:
        """Create or update DUT link (idempotent)"""

        # Check if link already exists
        existing = self.session.query(DUTLink).filter(
            and_(
                DUTLink.dut_uid == dut_uid,
                DUTLink.test_session_id == test_session_id,
                DUTLink.source_type == source_type,
                DUTLink.source_record_id == source_record_id
            )
        ).first()

        if existing:
            # Update existing link
            existing.die_x = die_x if die_x is not None else existing.die_x
            existing.die_y = die_y if die_y is not None else existing.die_y
            existing.normalized_x = normalized_x if normalized_x is not None else existing.normalized_x
            existing.normalized_y = normalized_y if normalized_y is not None else existing.normalized_y
            existing.updated_by = user
            return existing

        # Create new link
        link = DUTLink(
            dut_uid=dut_uid,
            test_session_id=test_session_id,
            source_type=source_type,
            source_record_id=source_record_id,
            die_x=die_x,
            die_y=die_y,
            normalized_x=normalized_x,
            normalized_y=normalized_y,
            site_num=site_num,
            stdf_specific_dut_serial=stdf_specific_dut_serial,
            wafer_id=wafer_id,
            supplier=supplier,
            product=product,
            lot_id=lot_id,
            created_by=user,
            updated_by=user
        )
        self.session.add(link)
        self.session.flush()
        return link

    def get_related_data(
        self,
        dut_uid: str,
        include_stale: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get all related data for a DUT across sources

        Returns:
            Dict with keys: 'stdf_parametric', 'part_summary', 'wafer_die', etc.
            Each value is a list of dicts with source info
        """
        links = self.session.query(DUTLink).filter(
            DUTLink.dut_uid == dut_uid
        ).all()

        results: Dict[str, List[Dict[str, Any]]] = {
            'stdf_parametric': [],
            'part_summary': [],
            'wafer_die': [],
            'image': [],
            'log': []
        }

        for link in links:
            link_data = {
                'test_session_id': link.test_session_id,
                'source_record_id': link.source_record_id,
                'die_x': link.die_x,
                'die_y': link.die_y,
                'normalized_x': link.normalized_x,
                'normalized_y': link.normalized_y,
                'site_num': link.site_num,
                'stdf_specific_dut_serial': link.stdf_specific_dut_serial,
                'wafer_id': link.wafer_id,
                'supplier': link.supplier,
                'product': link.product,
                'lot_id': link.lot_id
            }
            results[link.source_type.value].append(link_data)

        return results

    def find_dut_by_identifiers(
        self,
        supplier: Optional[str] = None,
        product: Optional[str] = None,
        lot_id: Optional[str] = None,
        wafer_id: Optional[str] = None,
        die_x: Optional[int] = None,
        die_y: Optional[int] = None,
        stdf_specific_dut_serial: Optional[str] = None
    ) -> List[str]:
        """Find DUT UIDs matching given identifiers"""

        filters = []
        if supplier:
            filters.append(DUTLink.supplier == supplier)
        if product:
            filters.append(DUTLink.product == product)
        if lot_id:
            filters.append(DUTLink.lot_id == lot_id)
        if wafer_id:
            filters.append(DUTLink.wafer_id == wafer_id)
        if die_x is not None:
            filters.append(DUTLink.die_x == die_x)
        if die_y is not None:
            filters.append(DUTLink.die_y == die_y)
        if stdf_specific_dut_serial:
            filters.append(DUTLink.stdf_specific_dut_serial == stdf_specific_dut_serial)

        if not filters:
            raise ValueError("At least one identifier must be provided")

        results = self.session.query(DUTLink.dut_uid).filter(
            and_(*filters)
        ).distinct().all()

        return [r[0] for r in results]
```

---

### 2.2 Query/Read APIs

**New File:** `SA_15_production_data_storage/src/queries.py`

```python
"""Query helpers for normalized coordinates and DUT data"""

from typing import List, Dict, Any, Optional
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from .model import (
    STDFParametricData,
    WaferLevelDie,
    PartSummary,
    TestSession,
    OrientationTransform,
    DUTLink
)
from .transform_service import TransformService, DUTLinkService


def get_parametric_data_with_normalized_coords(
    session: Session,
    test_session_id: int,
    parameter_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch parametric data with normalized coordinates applied on-the-fly

    Args:
        session: Database session
        test_session_id: Test session ID
        parameter_name: Optional filter for specific parameter

    Returns:
        List of dicts with parametric data plus normalized_x, normalized_y
    """
    transform_service = TransformService(session)

    # Get transform for session (or identity)
    transform = transform_service.get_or_create_identity_transform(test_session_id)

    # Query parametric data with DUT info
    query = session.query(
        STDFParametricData,
        WaferLevelDie.raw_stdf_wafer_x_coord,
        WaferLevelDie.raw_stdf_wafer_y_coord,
        WaferLevelDie.wafer_id
    ).join(
        WaferLevelDie,
        STDFParametricData.dut_id == WaferLevelDie.id
    ).filter(
        STDFParametricData.test_session_id == test_session_id
    )

    if parameter_name:
        query = query.filter(STDFParametricData.parameter_name == parameter_name)

    results = []
    for param_data, raw_x, raw_y, wafer_id in query.all():
        if raw_x is not None and raw_y is not None:
            norm_x, norm_y = transform_service.apply_transform(raw_x, raw_y, transform)
        else:
            norm_x, norm_y = None, None

        results.append({
            'id': param_data.id,
            'test_session_id': param_data.test_session_id,
            'parameter_name': param_data.parameter_name,
            'value': param_data.value,
            'units': param_data.units,
            'raw_x': raw_x,
            'raw_y': raw_y,
            'normalized_x': norm_x,
            'normalized_y': norm_y,
            'wafer_id': wafer_id,
            'site_num': param_data.site_num,
            'passed': param_data.passed
        })

    return results


def get_cross_session_data_for_dut(
    session: Session,
    dut_uid: str
) -> Dict[str, Any]:
    """
    Get all parametric and summary data for a DUT across all sessions

    Args:
        session: Database session
        dut_uid: Canonical DUT identifier

    Returns:
        Dict with test sessions and associated data
    """
    link_service = DUTLinkService(session)

    # Get all links for this DUT
    related_data = link_service.get_related_data(dut_uid)

    # For each parametric link, fetch actual test data
    parametric_details = []
    for link_info in related_data['stdf_parametric']:
        if link_info['source_record_id']:
            param = session.query(STDFParametricData).get(link_info['source_record_id'])
            if param:
                parametric_details.append({
                    'parameter_name': param.parameter_name,
                    'value': param.value,
                    'units': param.units,
                    'passed': param.passed,
                    'test_session_id': param.test_session_id,
                    'normalized_x': link_info['normalized_x'],
                    'normalized_y': link_info['normalized_y']
                })

    return {
        'dut_uid': dut_uid,
        'parametric_tests': parametric_details,
        'part_summaries': related_data['part_summary'],
        'sessions_count': len(set(p['test_session_id'] for p in parametric_details))
    }
```

---

## Phase 3: Ingester Integration

### 3.1 Ingester Changes

**File:** `SA-21_production_data_ingester/src/stdf.py`

Add transform detection and DUT linking to the existing `stdf_to_db_pending` function:

```python
# Add to imports
from SA_15_production_data_storage.src.transform_service import (
    TransformService,
    DUTLinkService,
    TransformSource,
    TransformConfidence
)
from SA_15_production_data_storage.src.model import (
    OrientationTransform,
    DUTLink,
    SourceType
)

def detect_orientation_from_stdf(stdf: STDFParser) -> dict:
    """
    Attempt to detect orientation transform from STDF headers

    Returns dict with rotation_degrees, confidence, notes
    """
    # Check for orientation hints in STDF records
    # WIR (Wafer Information Record) or custom fields
    # This is supplier/tool specific - implement based on actual STDF contents

    # Example heuristic (customize based on actual data):
    # - Check wafer flat orientation field if present
    # - Check prober configuration strings
    # - Look for rotation metadata in job_name or node_name

    # For now, default to identity
    return {
        'rotation_degrees': 0.0,
        'translation_x': 0.0,
        'translation_y': 0.0,
        'flip_x': False,
        'flip_y': False,
        'scale': 1.0,
        'source': TransformSource.auto_header,
        'confidence': TransformConfidence.unknown,
        'notes': 'Auto-detected from STDF headers (no orientation found, using identity)'
    }

def create_orientation_transform(
    session: Session,
    test_session: TestSession,
    stdf: STDFParser
) -> OrientationTransform:
    """Create orientation transform for test session during ingest"""

    transform_service = TransformService(session)

    # Attempt auto-detection
    detected = detect_orientation_from_stdf(stdf)

    logger.info(
        f"Creating orientation transform for session {test_session.id}",
        extra={
            'rotation': detected['rotation_degrees'],
            'confidence': detected['confidence'].value,
            'source': detected['source'].value
        }
    )

    # Create transform
    transform = transform_service.upsert_transform(
        test_session_id=test_session.id,
        rotation_degrees=detected['rotation_degrees'],
        translation_x=detected['translation_x'],
        translation_y=detected['translation_y'],
        flip_x=detected['flip_x'],
        flip_y=detected['flip_y'],
        scale=detected['scale'],
        source=detected['source'],
        confidence=detected['confidence'],
        notes=detected['notes'],
        user='ingester_auto'
    )

    return transform

def create_dut_links_for_session(
    session: Session,
    test_session: TestSession,
    stdf: STDFParser
) -> int:
    """Create DUT links for all devices in session"""

    transform_service = TransformService(session)
    link_service = DUTLinkService(session)

    # Get transform for normalized coordinates
    transform = transform_service.get_or_create_identity_transform(test_session.id)

    # Extract supplier/product from test_session or STDF
    # (Adapt based on where this info lives in your STDF)
    supplier = stdf.node_name.split('_')[0] if '_' in stdf.node_name else stdf.node_name
    product = stdf.job_id.split('_')[0] if '_' in stdf.job_id else stdf.job_id
    lot_id = test_session.lot_id

    links_created = 0

    # Link WaferLevelDie records
    wafer_dies = session.query(WaferLevelDie).filter(
        WaferLevelDie.test_session_id == test_session.id
    ).all()

    for die in wafer_dies:
        if die.raw_stdf_wafer_x_coord is not None and die.raw_stdf_wafer_y_coord is not None:
            # Compute normalized coordinates
            norm_x, norm_y = transform_service.apply_transform(
                die.raw_stdf_wafer_x_coord,
                die.raw_stdf_wafer_y_coord,
                transform
            )

            # Generate canonical UID
            dut_uid = link_service.generate_dut_uid(
                supplier=supplier,
                product=product,
                lot_id=lot_id,
                wafer_id=die.wafer_id or 'unknown',
                normalized_x=norm_x,
                normalized_y=norm_y
            )

            # Create link
            link_service.create_dut_link(
                dut_uid=dut_uid,
                test_session_id=test_session.id,
                source_type=SourceType.wafer_die,
                source_record_id=die.id,
                die_x=die.raw_stdf_wafer_x_coord,
                die_y=die.raw_stdf_wafer_y_coord,
                normalized_x=norm_x,
                normalized_y=norm_y,
                wafer_id=die.wafer_id,
                supplier=supplier,
                product=product,
                lot_id=lot_id,
                user='ingester_auto'
            )
            links_created += 1

    # Link STDFParametricData records (via their DUT association)
    # This could be done in bulk after parametric data insertion

    logger.info(f"Created {links_created} DUT links for session {test_session.id}")
    return links_created

# Modify existing stdf_to_db_pending to call these functions:
# Add after test_session creation and flush:
#
#   # Create orientation transform
#   create_orientation_transform(session, test_session, stdf)
#
# Add after all DUT and parametric data insertion:
#
#   # Create DUT cross-links
#   create_dut_links_for_session(session, test_session, stdf)
```

**Integration Points:**

1. After `TestSession` creation in `stdf_to_db_pending`, call `create_orientation_transform()`
2. After all DUT and parametric data commits, call `create_dut_links_for_session()`
3. Add structured logging for transform creation and link counts
4. Add metrics emission (if using CloudWatch/Prometheus)

---

### 3.2 Observability

Add metrics and logs:

```python
# In handler.py or stdf.py, add:

logger.info(
    "orientation_transform_created",
    extra={
        'test_session_id': test_session.id,
        'rotation_degrees': transform.rotation_degrees,
        'source': transform.source.value,
        'confidence': transform.confidence.value
    }
)

logger.warning(
    "missing_orientation_metadata",
    extra={
        'test_session_id': test_session.id,
        'lot_id': test_session.lot_id,
        'message': 'No orientation found in STDF, using identity transform'
    }
)

logger.info(
    "dut_links_created",
    extra={
        'test_session_id': test_session.id,
        'link_count': links_created
    }
)
```

---

## Phase 4: CLI/Admin Utilities

### 4.1 CLI Module

**New File:** `SA_15_production_data_storage/cli_transform.py`

```python
#!/usr/bin/env python3
"""CLI utilities for managing coordinate transforms"""

import argparse
import sys
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import os

from src.model import TestSession, OrientationTransform
from src.transform_service import TransformService, DUTLinkService, TransformSource, TransformConfidence


def get_session() -> Session:
    """Get database session from environment"""
    sql_url = os.getenv('SQLALCHEMY_URL')
    if not sql_url:
        raise ValueError("SQLALCHEMY_URL environment variable not set")
    engine = create_engine(sql_url)
    return Session(engine)


def cmd_create_transform(args):
    """Create or update a transform for a test session"""
    session = get_session()
    transform_service = TransformService(session)

    try:
        transform = transform_service.upsert_transform(
            test_session_id=args.test_session_id,
            rotation_degrees=args.rotation,
            translation_x=args.translate_x,
            translation_y=args.translate_y,
            flip_x=args.flip_x,
            flip_y=args.flip_y,
            scale=args.scale,
            source=TransformSource[args.source],
            confidence=TransformConfidence[args.confidence],
            notes=args.notes,
            user=args.user or 'cli_admin'
        )
        session.commit()
        print(f"✓ Created transform ID {transform.id} (version {transform.version}) for session {args.test_session_id}")
        print(f"  Rotation: {transform.rotation_degrees}°, Translation: ({transform.translation_x}, {transform.translation_y})")
    except Exception as e:
        session.rollback()
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        session.close()


def cmd_preview_transform(args):
    """Preview the effect of a transform on sample coordinates"""
    session = get_session()
    transform_service = TransformService(session)

    try:
        transform = session.query(OrientationTransform).filter(
            OrientationTransform.test_session_id == args.test_session_id,
            OrientationTransform.is_active == True
        ).first()

        if not transform:
            print(f"No active transform found for session {args.test_session_id}")
            return

        print(f"\nTransform for session {args.test_session_id} (version {transform.version}):")
        print(f"  Rotation: {transform.rotation_degrees}°")
        print(f"  Translation: ({transform.translation_x}, {transform.translation_y})")
        print(f"  Flips: X={transform.flip_x}, Y={transform.flip_y}")
        print(f"  Scale: {transform.scale}")
        print(f"  Confidence: {transform.confidence.value}\n")

        # Apply to sample points
        test_points = [
            (0, 0),
            (10, 0),
            (0, 10),
            (10, 10),
            (-5, -5)
        ]

        print("Sample transformations:")
        print(f"{'Raw X':<10} {'Raw Y':<10} {'Normalized X':<15} {'Normalized Y':<15}")
        print("-" * 50)
        for x, y in test_points:
            norm_x, norm_y = transform_service.apply_transform(x, y, transform)
            print(f"{x:<10} {y:<10} {norm_x:<15.2f} {norm_y:<15.2f}")

    finally:
        session.close()


def cmd_query_dut(args):
    """Query all related data for a DUT"""
    session = get_session()
    link_service = DUTLinkService(session)

    try:
        # If given identifiers instead of UID, find the UID first
        if args.dut_uid:
            dut_uids = [args.dut_uid]
        else:
            dut_uids = link_service.find_dut_by_identifiers(
                supplier=args.supplier,
                product=args.product,
                lot_id=args.lot_id,
                wafer_id=args.wafer_id,
                die_x=args.die_x,
                die_y=args.die_y
            )

        if not dut_uids:
            print("No DUTs found matching criteria")
            return

        for dut_uid in dut_uids:
            print(f"\n{'=' * 80}")
            print(f"DUT UID: {dut_uid}")
            print(f"{'=' * 80}")

            related = link_service.get_related_data(dut_uid)

            for source_type, records in related.items():
                if records:
                    print(f"\n{source_type.upper()} ({len(records)} records):")
                    for i, rec in enumerate(records[:5], 1):  # Show first 5
                        print(f"  [{i}] Session {rec['test_session_id']}: "
                              f"({rec['die_x']}, {rec['die_y']}) → "
                              f"({rec['normalized_x']:.2f}, {rec['normalized_y']:.2f})")
                    if len(records) > 5:
                        print(f"  ... and {len(records) - 5} more")

    finally:
        session.close()


def cmd_backfill_links(args):
    """Backfill DUT links for existing sessions"""
    session = get_session()
    transform_service = TransformService(session)
    link_service = DUTLinkService(session)

    try:
        # Query sessions in date range
        query = session.query(TestSession)
        if args.start_date:
            start = datetime.fromisoformat(args.start_date)
            query = query.filter(TestSession.start_time >= start)
        if args.end_date:
            end = datetime.fromisoformat(args.end_date)
            query = query.filter(TestSession.start_time <= end)

        sessions = query.all()
        print(f"Found {len(sessions)} sessions to backfill")

        for i, test_session in enumerate(sessions, 1):
            print(f"\r[{i}/{len(sessions)}] Processing session {test_session.id}...", end='')

            # Ensure transform exists
            transform_service.get_or_create_identity_transform(
                test_session.id,
                source=TransformSource.manual,
                notes="Backfill - identity transform"
            )

            # Create links (simplified - adapt from ingester logic)
            # TODO: Implement full backfill logic similar to create_dut_links_for_session

            if i % 100 == 0:
                session.commit()

        session.commit()
        print(f"\n✓ Backfill complete")

    except Exception as e:
        session.rollback()
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description='Coordinate Transform Management CLI')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # create-transform
    create_parser = subparsers.add_parser('create-transform', help='Create/update a coordinate transform')
    create_parser.add_argument('test_session_id', type=int, help='Test session ID')
    create_parser.add_argument('--rotation', type=float, default=0.0, help='Rotation in degrees (0-360)')
    create_parser.add_argument('--translate-x', type=float, default=0.0, help='X translation')
    create_parser.add_argument('--translate-y', type=float, default=0.0, help='Y translation')
    create_parser.add_argument('--flip-x', action='store_true', help='Flip X axis')
    create_parser.add_argument('--flip-y', action='store_true', help='Flip Y axis')
    create_parser.add_argument('--scale', type=float, default=1.0, help='Scale factor')
    create_parser.add_argument('--source', choices=['manual', 'auto_header', 'prober_recipe', 'config'], default='manual')
    create_parser.add_argument('--confidence', choices=['high', 'medium', 'low', 'unknown'], default='medium')
    create_parser.add_argument('--notes', type=str, help='Notes about this transform')
    create_parser.add_argument('--user', type=str, help='User creating transform')
    create_parser.set_defaults(func=cmd_create_transform)

    # preview-transform
    preview_parser = subparsers.add_parser('preview', help='Preview transform on sample coordinates')
    preview_parser.add_argument('test_session_id', type=int, help='Test session ID')
    preview_parser.set_defaults(func=cmd_preview_transform)

    # query-dut
    query_parser = subparsers.add_parser('query-dut', help='Query all data for a DUT')
    query_parser.add_argument('--dut-uid', type=str, help='DUT UID to query')
    query_parser.add_argument('--supplier', type=str, help='Supplier filter')
    query_parser.add_argument('--product', type=str, help='Product filter')
    query_parser.add_argument('--lot-id', type=str, help='Lot ID filter')
    query_parser.add_argument('--wafer-id', type=str, help='Wafer ID filter')
    query_parser.add_argument('--die-x', type=int, help='Die X coordinate filter')
    query_parser.add_argument('--die-y', type=int, help='Die Y coordinate filter')
    query_parser.set_defaults(func=cmd_query_dut)

    # backfill-links
    backfill_parser = subparsers.add_parser('backfill', help='Backfill DUT links for existing sessions')
    backfill_parser.add_argument('--start-date', type=str, help='Start date (ISO format)')
    backfill_parser.add_argument('--end-date', type=str, help='End date (ISO format)')
    backfill_parser.set_defaults(func=cmd_backfill_links)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
```

**Usage Examples:**

```bash
# Create a 90° rotation transform
python cli_transform.py create-transform 12345 --rotation 90 --source manual --confidence high --notes "Correcting prober orientation"

# Preview transform effect
python cli_transform.py preview 12345

# Query DUT by coordinates
python cli_transform.py query-dut --lot-id LOT123 --wafer-id W01 --die-x 5 --die-y 10

# Backfill links for date range
python cli_transform.py backfill --start-date 2025-01-01 --end-date 2025-01-31
```

---

## Phase 5: Testing & Validation

### 5.1 Unit Tests

**New File:** `SA_15_production_data_storage/tests/test_transform_service.py`

```python
"""Unit tests for coordinate transforms"""

import pytest
import math
from src.transform_service import TransformService
from src.model import OrientationTransform, TransformSource, TransformConfidence


class TestTransformMath:
    """Test coordinate transformation math"""

    def test_identity_transform(self):
        """Identity transform should not change coordinates"""
        transform = OrientationTransform(
            test_session_id=1,
            rotation_degrees=0.0,
            translation_x=0.0,
            translation_y=0.0,
            flip_x=False,
            flip_y=False,
            scale=1.0,
            source=TransformSource.manual,
            confidence=TransformConfidence.high,
            created_by='test',
            updated_by='test'
        )

        service = TransformService(None)  # No session needed for pure math
        x, y = service.apply_transform(10, 20, transform)

        assert x == 10.0
        assert y == 20.0

    def test_rotation_90_degrees(self):
        """90° CCW rotation should swap and negate"""
        transform = OrientationTransform(
            test_session_id=1,
            rotation_degrees=90.0,
            translation_x=0.0,
            translation_y=0.0,
            flip_x=False,
            flip_y=False,
            scale=1.0,
            source=TransformSource.manual,
            confidence=TransformConfidence.high,
            created_by='test',
            updated_by='test'
        )

        service = TransformService(None)

        # (1, 0) → (0, 1)
        x, y = service.apply_transform(1, 0, transform)
        assert abs(x - 0.0) < 0.001
        assert abs(y - 1.0) < 0.001

        # (0, 1) → (-1, 0)
        x, y = service.apply_transform(0, 1, transform)
        assert abs(x - (-1.0)) < 0.001
        assert abs(y - 0.0) < 0.001

    def test_rotation_180_degrees(self):
        """180° rotation should negate both coordinates"""
        transform = OrientationTransform(
            test_session_id=1,
            rotation_degrees=180.0,
            translation_x=0.0,
            translation_y=0.0,
            flip_x=False,
            flip_y=False,
            scale=1.0,
            source=TransformSource.manual,
            confidence=TransformConfidence.high,
            created_by='test',
            updated_by='test'
        )

        service = TransformService(None)
        x, y = service.apply_transform(5, 10, transform)

        assert abs(x - (-5.0)) < 0.001
        assert abs(y - (-10.0)) < 0.001

    def test_translation(self):
        """Translation should offset coordinates"""
        transform = OrientationTransform(
            test_session_id=1,
            rotation_degrees=0.0,
            translation_x=100.0,
            translation_y=200.0,
            flip_x=False,
            flip_y=False,
            scale=1.0,
            source=TransformSource.manual,
            confidence=TransformConfidence.high,
            created_by='test',
            updated_by='test'
        )

        service = TransformService(None)
        x, y = service.apply_transform(10, 20, transform)

        assert x == 110.0
        assert y == 220.0

    def test_flip_x(self):
        """Flip X should negate X coordinate"""
        transform = OrientationTransform(
            test_session_id=1,
            rotation_degrees=0.0,
            translation_x=0.0,
            translation_y=0.0,
            flip_x=True,
            flip_y=False,
            scale=1.0,
            source=TransformSource.manual,
            confidence=TransformConfidence.high,
            created_by='test',
            updated_by='test'
        )

        service = TransformService(None)
        x, y = service.apply_transform(10, 20, transform)

        assert x == -10.0
        assert y == 20.0

    def test_scale(self):
        """Scale should multiply coordinates"""
        transform = OrientationTransform(
            test_session_id=1,
            rotation_degrees=0.0,
            translation_x=0.0,
            translation_y=0.0,
            flip_x=False,
            flip_y=False,
            scale=2.5,
            source=TransformSource.manual,
            confidence=TransformConfidence.high,
            created_by='test',
            updated_by='test'
        )

        service = TransformService(None)
        x, y = service.apply_transform(10, 20, transform)

        assert x == 25.0
        assert y == 50.0

    def test_composition_scale_rotate_translate(self):
        """Test order of operations: scale → flip → rotate → translate"""
        transform = OrientationTransform(
            test_session_id=1,
            rotation_degrees=90.0,
            translation_x=10.0,
            translation_y=20.0,
            flip_x=False,
            flip_y=False,
            scale=2.0,
            source=TransformSource.manual,
            confidence=TransformConfidence.high,
            created_by='test',
            updated_by='test'
        )

        service = TransformService(None)

        # Start: (1, 0)
        # After scale (2.0): (2, 0)
        # After 90° rotation: (0, 2)
        # After translation: (10, 22)
        x, y = service.apply_transform(1, 0, transform)

        assert abs(x - 10.0) < 0.001
        assert abs(y - 22.0) < 0.001

    def test_round_trip_with_inverse(self):
        """Applying transform and its inverse should return original coords"""
        # Forward: 45° rotation
        forward = OrientationTransform(
            test_session_id=1,
            rotation_degrees=45.0,
            translation_x=0.0,
            translation_y=0.0,
            flip_x=False,
            flip_y=False,
            scale=1.0,
            source=TransformSource.manual,
            confidence=TransformConfidence.high,
            created_by='test',
            updated_by='test'
        )

        # Inverse: -45° rotation (or 315°)
        inverse = OrientationTransform(
            test_session_id=1,
            rotation_degrees=315.0,
            translation_x=0.0,
            translation_y=0.0,
            flip_x=False,
            flip_y=False,
            scale=1.0,
            source=TransformSource.manual,
            confidence=TransformConfidence.high,
            created_by='test',
            updated_by='test'
        )

        service = TransformService(None)

        orig_x, orig_y = 10.0, 20.0
        transformed_x, transformed_y = service.apply_transform(orig_x, orig_y, forward)
        final_x, final_y = service.apply_transform(transformed_x, transformed_y, inverse)

        assert abs(final_x - orig_x) < 0.001
        assert abs(final_y - orig_y) < 0.001


class TestDUTLinkService:
    """Test DUT linking logic"""

    def test_dut_uid_generation_deterministic(self):
        """Same inputs should always generate same UID"""
        from src.transform_service import DUTLinkService

        service = DUTLinkService(None)

        uid1 = service.generate_dut_uid("SUPPLIER", "PROD", "LOT123", "W01", 10.5, 20.3)
        uid2 = service.generate_dut_uid("SUPPLIER", "PROD", "LOT123", "W01", 10.5, 20.3)

        assert uid1 == uid2

    def test_dut_uid_different_for_different_coords(self):
        """Different coordinates should generate different UIDs"""
        from src.transform_service import DUTLinkService

        service = DUTLinkService(None)

        uid1 = service.generate_dut_uid("SUPPLIER", "PROD", "LOT123", "W01", 10.5, 20.3)
        uid2 = service.generate_dut_uid("SUPPLIER", "PROD", "LOT123", "W01", 10.6, 20.3)

        assert uid1 != uid2

    def test_dut_uid_case_insensitive(self):
        """UID should be case-insensitive for supplier/product/lot"""
        from src.transform_service import DUTLinkService

        service = DUTLinkService(None)

        uid1 = service.generate_dut_uid("supplier", "product", "lot123", "w01", 10.5, 20.3)
        uid2 = service.generate_dut_uid("SUPPLIER", "PRODUCT", "LOT123", "W01", 10.5, 20.3)

        assert uid1 == uid2
```

---

### 5.2 Integration Tests

**New File:** `SA_15_production_data_storage/tests/test_integration_transform.py`

```python
"""Integration tests for transform + ingestion flow"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.model import (
    Base,
    TestSession,
    SourceFile,
    WaferLevelDie,
    OrientationTransform,
    DUTLink,
    SourceType
)
from src.transform_service import TransformService, DUTLinkService, TransformSource, TransformConfidence


@pytest.fixture
def db_session():
    """Create in-memory test database"""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    session = Session(engine)
    yield session
    session.close()


def test_two_sessions_same_wafer_different_orientations(db_session):
    """
    Integration test: Ingest same wafer with two different orientations,
    verify normalized coordinates align
    """

    # Setup: Create source files and test sessions
    now = datetime.now(timezone.utc)

    source1 = SourceFile(
        object_key='test1.stdf',
        version_key='v1',
        parse_start_time=now,
        parse_complete=True,
        created_by='test',
        updated_by='test'
    )
    db_session.add(source1)
    db_session.flush()

    session1 = TestSession(
        source_file_id=source1.id,
        setup_time=now,
        start_time=now,
        end_time=now,
        test_mode_code='P',
        lot_id='LOT123',
        node_name='NODE1',
        tester_type='T1',
        job_name='JOB1',
        operator_id='OP1',
        tester_software_type='SW1',
        tester_software_version='1.0',
        created_by='test',
        updated_by='test'
    )
    db_session.add(session1)
    db_session.flush()

    source2 = SourceFile(
        object_key='test2.stdf',
        version_key='v2',
        parse_start_time=now,
        parse_complete=True,
        created_by='test',
        updated_by='test'
    )
    db_session.add(source2)
    db_session.flush()

    session2 = TestSession(
        source_file_id=source2.id,
        setup_time=now,
        start_time=now,
        end_time=now,
        test_mode_code='P',
        lot_id='LOT123',
        node_name='NODE1',
        tester_type='T1',
        job_name='JOB1',
        operator_id='OP1',
        tester_software_type='SW1',
        tester_software_version='1.0',
        created_by='test',
        updated_by='test'
    )
    db_session.add(session2)
    db_session.flush()

    # Create transforms
    transform_service = TransformService(db_session)

    # Session 1: Identity (0° rotation)
    t1 = transform_service.upsert_transform(
        test_session_id=session1.id,
        rotation_degrees=0.0,
        source=TransformSource.manual,
        confidence=TransformConfidence.high,
        notes="Reference orientation",
        user='test'
    )

    # Session 2: 90° rotation (wafer rotated CCW)
    t2 = transform_service.upsert_transform(
        test_session_id=session2.id,
        rotation_degrees=90.0,
        source=TransformSource.manual,
        confidence=TransformConfidence.high,
        notes="Rotated 90° CCW",
        user='test'
    )

    # Create DUTs for session 1
    # Die at (10, 0) in raw coords
    die1_s1 = WaferLevelDie(
        test_session_id=session1.id,
        wafer_id='W01',
        raw_stdf_wafer_x_coord=10,
        raw_stdf_wafer_y_coord=0,
        created_by='test',
        updated_by='test'
    )
    db_session.add(die1_s1)
    db_session.flush()

    # Create DUTs for session 2
    # Same physical die, but now at (0, 10) due to 90° rotation
    # After transform, should normalize to (10, 0) → rotated 90° → (0, 10)
    # Wait, let's think: if session1 is reference at (10, 0),
    # and session2 has wafer rotated 90° CCW, the same physical location
    # would be read as (0, 10) in session2's raw coords.
    # After applying 90° rotation to (0, 10): should give us back (10, 0) approximately

    # Actually, to test this properly:
    # Physical die location: (10, 0) in normalized space
    # Session 1 (0° rotation): reads as (10, 0) raw
    # Session 2 (90° CW rotation): reads as (0, -10) raw, needs -90° transform (or 270°)
    # Let's use inverse for session2:

    die1_s2 = WaferLevelDie(
        test_session_id=session2.id,
        wafer_id='W01',
        raw_stdf_wafer_x_coord=0,
        raw_stdf_wafer_y_coord=-10,
        created_by='test',
        updated_by='test'
    )
    db_session.add(die1_s2)
    db_session.flush()

    # Apply transforms and create links
    link_service = DUTLinkService(db_session)

    # Session 1 die
    norm_x1, norm_y1 = transform_service.apply_transform(10, 0, t1)
    dut_uid1 = link_service.generate_dut_uid(
        "SUPPLIER", "PRODUCT", "LOT123", "W01", norm_x1, norm_y1
    )
    link1 = link_service.create_dut_link(
        dut_uid=dut_uid1,
        test_session_id=session1.id,
        source_type=SourceType.wafer_die,
        source_record_id=die1_s1.id,
        die_x=10,
        die_y=0,
        normalized_x=norm_x1,
        normalized_y=norm_y1,
        wafer_id='W01',
        supplier='SUPPLIER',
        product='PRODUCT',
        lot_id='LOT123',
        user='test'
    )

    # Session 2 die (with inverse rotation to normalize)
    # Update transform for session 2 to be -90° (270°) to undo the physical rotation
    t2_corrected = transform_service.upsert_transform(
        test_session_id=session2.id,
        rotation_degrees=270.0,  # Inverse of 90° CCW
        source=TransformSource.manual,
        confidence=TransformConfidence.high,
        notes="Inverse rotation to normalize",
        user='test'
    )

    norm_x2, norm_y2 = transform_service.apply_transform(0, -10, t2_corrected)
    dut_uid2 = link_service.generate_dut_uid(
        "SUPPLIER", "PRODUCT", "LOT123", "W01", norm_x2, norm_y2
    )
    link2 = link_service.create_dut_link(
        dut_uid=dut_uid2,
        test_session_id=session2.id,
        source_type=SourceType.wafer_die,
        source_record_id=die1_s2.id,
        die_x=0,
        die_y=-10,
        normalized_x=norm_x2,
        normalized_y=norm_y2,
        wafer_id='W01',
        supplier='SUPPLIER',
        product='PRODUCT',
        lot_id='LOT123',
        user='test'
    )

    db_session.commit()

    # Verify: Both sessions should produce the same normalized coords and DUT UID
    assert abs(norm_x1 - norm_x2) < 0.1, f"Normalized X mismatch: {norm_x1} vs {norm_x2}"
    assert abs(norm_y1 - norm_y2) < 0.1, f"Normalized Y mismatch: {norm_y1} vs {norm_y2}"
    assert dut_uid1 == dut_uid2, "DUT UIDs should match for same physical device"

    # Query by DUT UID should return both sessions
    related = link_service.get_related_data(dut_uid1)
    assert len(related['wafer_die']) == 2, "Should find 2 wafer_die records for this DUT"

    session_ids = {r['test_session_id'] for r in related['wafer_die']}
    assert session_ids == {session1.id, session2.id}, "Should link both sessions"

    print(f"✓ Integration test passed: normalized coords aligned across rotated sessions")
    print(f"  Session 1: ({die1_s1.raw_stdf_wafer_x_coord}, {die1_s1.raw_stdf_wafer_y_coord}) → ({norm_x1:.2f}, {norm_y1:.2f})")
    print(f"  Session 2: ({die1_s2.raw_stdf_wafer_x_coord}, {die1_s2.raw_stdf_wafer_y_coord}) → ({norm_x2:.2f}, {norm_y2:.2f})")
    print(f"  DUT UID: {dut_uid1}")
```

---

### 5.3 Performance Tests

**Test Goals:**
- Lookups by `(supplier, product, lot, wafer, session_id)` < 50ms P95
- DUT UID lookups < 50ms P95
- Transform application on 1M coordinates < 5 seconds

**File:** `SA_15_production_data_storage/tests/test_performance.py`

```python
"""Performance tests for coordinate transforms and lookups"""

import pytest
import time
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.model import Base, OrientationTransform, DUTLink
from src.transform_service import TransformService


def test_transform_application_performance():
    """Benchmark transform application on 1M coordinates"""

    transform = OrientationTransform(
        test_session_id=1,
        rotation_degrees=45.0,
        translation_x=100.0,
        translation_y=200.0,
        flip_x=False,
        flip_y=False,
        scale=1.0,
        source='manual',
        confidence='high',
        created_by='test',
        updated_by='test'
    )

    service = TransformService(None)

    # Generate 1M random coordinates
    import random
    coords = [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(1_000_000)]

    start = time.time()
    for x, y in coords:
        service.apply_transform(x, y, transform)
    elapsed = time.time() - start

    print(f"\nTransformed 1M coordinates in {elapsed:.2f}s ({elapsed*1000:.2f}ms avg per 1k)")
    assert elapsed < 5.0, "Should transform 1M coordinates in < 5 seconds"


@pytest.mark.skip(reason="Requires large dataset - run manually")
def test_dut_link_lookup_performance(db_session):
    """Benchmark DUT UID lookup with 1M links"""
    # TODO: Populate with 1M DUT links, then benchmark queries
    pass
```

---

## Phase 6: Documentation

### 6.1 Architecture Documentation

**New File:** `SA_15_production_data_storage/COORDINATE_TRANSFORM_ARCHITECTURE.md`

```markdown
# Coordinate Transform Architecture

## Overview

This system provides per-test-session coordinate transformations to normalize device coordinates across different wafer orientations and equipment setups. It also maintains cross-links (DUT links) to retrieve all related data for a given device across multiple test sessions and data sources.

## Problem Statement

STDF parametric data stores raw die coordinates (X, Y) from test equipment, but wafer orientation varies by:
- Supplier (different probing conventions)
- Equipment (different coordinate systems)
- Test session (wafer rotation/placement)
- Product/lot (process variations)

Without normalization, comparing the same physical die across sessions is impossible.

## Solution

### 1. Coordinate Frame Definitions

**Raw STDF Frame (`stdf_raw`):**
- Origin: Equipment-specific (typically bottom-left or center)
- Units: Integer die positions or micrometers
- Axes: Equipment convention (may be rotated/flipped)

**Normalized Frame (`normalized_wafer`):**
- Origin: Wafer center (0, 0)
- Units: Micrometers or consistent die units
- Axes: Standard semiconductor convention (X = right, Y = up, wafer flat at bottom)

### 2. Transformation Model

Each test session has an `OrientationTransform` defining:

- **Scale** (s): Unit conversion or die pitch scaling
- **Flip X/Y** (fx, fy): Axis reflection (1 or -1)
- **Rotation** (θ): Counter-clockwise angle in degrees
- **Translation** (tx, ty): Offset to align origins

**Order of operations:**
```
(x_norm, y_norm) = Translate(Rotate(Flip(Scale(x_raw, y_raw))))
```

**Mathematical formula:**
```python
# 1. Scale
x1 = x_raw * s
y1 = y_raw * s

# 2. Flip
x2 = x1 * (-1 if fx else 1)
y2 = y1 * (-1 if fy else 1)

# 3. Rotate (CCW positive)
θ_rad = θ * π / 180
x3 = x2 * cos(θ_rad) - y2 * sin(θ_rad)
y3 = x2 * sin(θ_rad) + y2 * cos(θ_rad)

# 4. Translate
x_norm = x3 + tx
y_norm = y3 + ty
```

### 3. DUT Linking Strategy

**Canonical DUT UID:**
- Generated from: `{supplier, product, lot_id, wafer_id, normalized_x, normalized_y}`
- Hashed with SHA256 for fixed-length identifier (64 chars)
- **Deterministic:** Same physical die always produces same UID

**DUT Link Records:**
- Map `dut_uid` → multiple source records (parametric data, part summaries, images, logs)
- Enable queries like "find all data for this device across all sessions"

### 4. Versioning & Provenance

**Transform Versioning:**
- Each update creates a new version, old versions are deactivated (`is_active=False`)
- Maintains lineage for audit and rollback
- Cache tables (e.g., `normalized_coordinates_cache`) track which transform version was used

**Provenance Fields:**
- `source`: How transform was determined (manual, auto-header, prober_recipe, config)
- `confidence`: Quality indicator (high/medium/low/unknown)
- `notes`: Human-readable context

## Workflows

### Workflow 1: Ingestion with Auto-Detection

```
1. Lambda receives S3 STDF file notification
2. Parse STDF headers for orientation hints (wafer flat, prober config)
3. Create/update TestSession
4. Detect or default to identity OrientationTransform
5. Parse parametric data and DUTs (WaferLevelDie records)
6. For each DUT:
   a. Apply transform to raw coordinates
   b. Generate canonical dut_uid from normalized coords
   c. Create DUTLink record
7. Commit transaction
8. Emit observability metrics (transform applied, link count)
```

### Workflow 2: Manual Transform Correction

```
1. Admin runs: `cli_transform.py create-transform <session_id> --rotation 90 --source manual`
2. System:
   a. Deactivates old transform (version N)
   b. Creates new transform (version N+1)
   c. Marks normalized_coordinates_cache as stale
3. Admin runs: `cli_transform.py preview <session_id>` to verify
4. Backfill normalized coordinates (or compute on-the-fly on next query)
```

### Workflow 3: Cross-Session DUT Query

```
1. User queries: "Get all parametric data for die at (10.5, 20.3) in LOT123/W01"
2. System:
   a. Generate dut_uid from query parameters
   b. Query DUTLink table for all records with this dut_uid
   c. For each link, fetch associated STDFParametricData, PartSummary, etc.
   d. Return unified view across all sessions
3. Result: User sees how this device performed across multiple test runs
```

## Database Schema

### Tables

- **`orientation_transform`**: Per-session transform metadata
- **`dut_link`**: Cross-reference DUT identifiers to source records
- **`normalized_coordinates_cache`** (optional): Materialized normalized coords

See `COORDINATE_TRANSFORM_IMPLEMENTATION_PLAN.md` for detailed schema definitions.

## Performance Considerations

- **Indexes:** Critical on `(test_session_id)`, `(dut_uid)`, `(supplier, product, lot_id, wafer_id, normalized_x, normalized_y)`
- **On-the-fly vs. Materialized:**
  - On-the-fly: More flexible, no storage overhead, ~1ms per transform
  - Materialized: Faster queries, requires cache invalidation on transform updates
- **Batch Operations:** Backfills and link creation should batch commits every 1000 records

## Testing Strategy

- **Unit Tests:** Transform math (rotation, flip, scale, composition, round-trip)
- **Integration Tests:** Two sessions, same wafer, different orientations → same normalized coords
- **Performance Tests:** 1M transforms < 5s, lookups < 50ms P95

## Rollout Plan

1. **Phase 1:** Deploy schema changes (migrations are non-breaking)
2. **Phase 2:** Enable transform creation in ingester (defaults to identity, logs warnings)
3. **Phase 3:** Enable DUT linking (parallel writes, no reads yet)
4. **Phase 4:** Backfill historical sessions with identity transforms
5. **Phase 5:** Enable normalized coordinate reads via feature flag
6. **Phase 6:** Train users on CLI tools, begin manual corrections
7. **Phase 7:** Optimize based on production metrics

## Future Enhancements

- **Auto-calibration:** Machine learning to detect orientation from parametric patterns
- **Spatial clustering:** Group devices by normalized location for yield analysis
- **Image alignment:** Use transform to align wafer map images with parametric data
- **Multi-pass transforms:** Handle retest scenarios where same wafer probed multiple times

## References

- STDF Specification: [IEEE 1445-1998](https://standards.ieee.org/)
- Coordinate Systems in Semiconductor Manufacturing: [SEMI Standard]
- Similar approaches: Yield Management Systems (YMS) from KLA, PDF Solutions
```

---

### 6.2 README Updates

**File:** `SA_15_production_data_storage/README.md` (append)

```markdown
## Coordinate Transforms

This repository now supports per-session coordinate transformations to normalize device coordinates across different wafer orientations.

### Quick Start

**Create a transform:**
```bash
python cli_transform.py create-transform 12345 --rotation 90 --source manual --notes "Corrected prober orientation"
```

**Preview transform:**
```bash
python cli_transform.py preview 12345
```

**Query DUT data:**
```bash
python cli_transform.py query-dut --lot-id LOT123 --wafer-id W01 --die-x 10 --die-y 20
```

See [COORDINATE_TRANSFORM_ARCHITECTURE.md](COORDINATE_TRANSFORM_ARCHITECTURE.md) for details.
```

---

**File:** `SA-21_production_data_ingester/README.md` (create or append)

```markdown
# SA-21 Production Data Ingester

Lambda function for ingesting STDF files from S3 into production database.

## Features

- Parses STDF files and extracts parametric data, DUT info, bin summaries
- **NEW:** Creates per-session coordinate transformations for wafer orientation normalization
- **NEW:** Generates DUT cross-links for querying devices across sessions
- Handles duplicate detection via source file version tracking
- Error notifications via email/SMS

## Environment Variables

- `SQLALCHEMY_URL`: Database connection string
- `ENABLE_MAIL_NOTIFIER`: Enable email notifications (optional)
- `ENABLE_SMS_NOTIFIER`: Enable SMS notifications (optional)

## Coordinate Transforms

The ingester automatically attempts to detect wafer orientation from STDF headers. If no orientation metadata is found, it defaults to an identity transform and logs a warning.

To manually correct transforms after ingest, use the CLI in `SA_15_production_data_storage`:
```bash
python cli_transform.py create-transform <session_id> --rotation <degrees>
```

## Development

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run tests:**
```bash
pytest tests/
```

**Deploy:**
```bash
serverless deploy --stage dev
```
```

---

## Phase 7: Acceptance Criteria Checklist

### Migrations
- [ ] `add_orientation_transform_table.py` migration runs cleanly (up and down)
- [ ] `add_dut_link_table.py` migration runs cleanly (up and down)
- [ ] Migrations are backward-compatible (no breaking changes to existing queries)
- [ ] All indexes and constraints created

### Data Access Layer
- [ ] `TransformService` implemented with `apply_transform`, `upsert_transform`, `get_or_create_identity_transform`
- [ ] `DUTLinkService` implemented with `generate_dut_uid`, `create_dut_link`, `get_related_data`
- [ ] Query helpers for normalized coordinates and cross-session data

### Ingester Integration
- [ ] Ingester creates `OrientationTransform` for each session (identity or detected)
- [ ] Ingester creates `DUTLink` records for all devices
- [ ] Structured logging for transform creation and missing metadata warnings
- [ ] Metrics emitted for link counts

### CLI Tools
- [ ] `create-transform` command works and validates inputs
- [ ] `preview` command shows sample transformations
- [ ] `query-dut` command retrieves cross-session data
- [ ] `backfill` command processes historical sessions
- [ ] Help text and usage examples documented

### Testing
- [ ] Unit tests pass for all rotation, flip, scale, translation combinations
- [ ] Unit tests verify DUT UID determinism
- [ ] Integration test: two sessions, same wafer, different orientations → same normalized coords
- [ ] Performance test: 1M transforms < 5 seconds
- [ ] Performance test: Lookups < 50ms P95 (on production-like dataset)

### Observability
- [ ] Structured logs for "transform applied", "missing transform", "link count"
- [ ] Logs include test_session_id, rotation, confidence, source
- [ ] Metrics emitted (CloudWatch or equivalent) for ingestion pipeline

### Documentation
- [ ] Architecture document written (transform math, coordinate frames, DUT linking)
- [ ] README updates in both repos
- [ ] Example workflows documented
- [ ] Rollback plan documented

### Validation
- [ ] Smoke test: Ingest sample STDF file, verify transform and links created
- [ ] End-to-end test: Query DUT by raw coordinates, retrieve data from two sessions
- [ ] Regression test: Existing STDF queries still work (no breaking changes)

---

## Phase 8: Rollout & Deployment

### Pre-Deployment

1. **Code Review:** All PRs reviewed and approved
2. **Staging Deployment:** Deploy to staging environment, run full test suite
3. **Performance Baseline:** Measure current query latencies for comparison
4. **Rollback Plan:** Document steps to revert migrations and code

### Deployment Sequence

#### Step 1: Schema Migration (SA_15_production_data_storage)
```bash
# Backup database first!
mysqldump -u user -p production_db > backup_$(date +%Y%m%d).sql

# Run migrations
export SQLALCHEMY_URL="mysql://..."
alembic upgrade head

# Verify tables created
mysql -u user -p -e "SHOW TABLES LIKE 'orientation_transform'"
mysql -u user -p -e "SHOW TABLES LIKE 'dut_link'"
```

**Rollback (if needed):**
```bash
alembic downgrade -1  # Or -2 for both migrations
```

#### Step 2: Deploy Storage Layer (SA_15_production_data_storage)
```bash
