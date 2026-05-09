//
// Created by igor on 3/9/26.
//

/*
 * Sweep SAH builder
 * 1) for each primitive compute: | DONE
 *      1.1) bounding box
 *      1.2) centroid
 * 2) compute bbox for all primitives - root node | DONE
 * 3) determine leaf conditions | DONE
 *      3.1) primitive count < treshold
 *      3.2) maximum tree depth?
 *    if leaf:
 *      3.3) mark node as leaf
 *      3.4) store primitives in node
 *      3.5) stop recursion
 * 4) evaluate split for each axis (x, y, z) | DONE
 *      4.1) sort primitives by centroid
 *      4.2) define candidate splits (every split between primitives)
 *      4.3) compute bboxes for all splits (left, right)
 *      4.4) evaluate SAH for each split:
 *              Cost = trav_cost + A_L / A_P * (N_L * intersect_cost) + A_R / A_P * (N_R * intersect_cost)
 * 5) compare leaf_cost and split_cost | DONE
 *      leaf_cost = primitives_in_node * intersection_cost
 *      if (best_split_cost > leaf_cost) create a leaf node
 * 6) divide primitives into left child primitives and right | DONE
 * 7) create child nodes
 * 8) recurse for left and right child
*/

/* BLAS - BVH, where leafs is triangles (nothing more)
 *
 * TLAS - BVH, where leafs is MeshInstance, where
 * MeshInstance is struct described below:
 * struct MeshInstance {
 *     BVH* bvh; //(or bvh index)
 *     Mat4 modelMatrix;
 * };
 */
