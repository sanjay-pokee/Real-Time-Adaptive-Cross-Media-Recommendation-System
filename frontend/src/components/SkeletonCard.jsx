/**
 * Placeholder for a RecommendationCard while a search is in flight.
 *
 * Mirrors that card's real geometry - a 144px media header, then the body -
 * rather than approximating it. A skeleton whose shape does not match what
 * replaces it makes the grid jump on every search, which reads as a bug even
 * though nothing is wrong.
 */
export default function SkeletonCard() {
  return (
    <div className="panel flex flex-col overflow-hidden">
      {/* media header — same h-36 as the card's backdrop strip */}
      <div className="skeleton h-36 w-full shrink-0" />

      <div className="flex flex-1 flex-col gap-3 p-4">
        <div className="flex flex-col gap-2">
          <div className="skeleton h-2.5 w-24 rounded" />
          <div className="skeleton h-2.5 w-1/2 rounded" />
        </div>

        <div className="flex flex-col gap-1.5">
          <div className="skeleton h-3 w-full rounded" />
          <div className="skeleton h-3 w-11/12 rounded" />
        </div>

        <div className="flex gap-1.5">
          <div className="skeleton h-5 w-16 rounded-full" />
          <div className="skeleton h-5 w-14 rounded-full" />
          <div className="skeleton h-5 w-20 rounded-full" />
        </div>

        <div className="mt-auto flex flex-col gap-2 border-t border-line pt-3.5">
          <div className="flex items-center justify-between">
            <div className="skeleton h-2.5 w-24 rounded" />
            <div className="skeleton h-3 w-10 rounded" />
          </div>
          <div className="skeleton h-2 w-full rounded-full" />
          <div className="flex gap-3">
            <div className="skeleton h-2.5 w-20 rounded" />
            <div className="skeleton h-2.5 w-16 rounded" />
          </div>
        </div>

        <div className="flex items-center gap-1 border-t border-line pt-3">
          {[0, 1, 2, 3, 4].map((index) => (
            <div key={index} className="skeleton h-7 w-7 rounded-lg" />
          ))}
          <div className="skeleton ml-auto h-3 w-20 rounded" />
        </div>
      </div>
    </div>
  );
}
